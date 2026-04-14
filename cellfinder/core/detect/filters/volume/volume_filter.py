import multiprocessing as mp
import os
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from brainglobe_utils.cells.cells import Cell
from tifffile import tifffile
from tqdm import tqdm

from cellfinder.core import logger, types
from cellfinder.core.detect.filters.plane import PlaneFilter
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter import BallFilter
from cellfinder.core.detect.filters.volume.laplacian_filter import (
    LaplacianFilter3D,
)
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)
from cellfinder.core.detect.filters.volume.threshold_filter import (
    ThresholdFilter3D,
)
from cellfinder.core.tools.threading import (
    EOFSignal,
    ProcessWithException,
    ThreadWithException,
)
from cellfinder.core.tools.tools import inference_wrapper

_done_signal = "is_done"


@inference_wrapper
def _plane_filter(
    process: ProcessWithException,
    plane_filter: PlaneFilter,
    n_threads: int,
    buffers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
):
    """
    When running on cpu, we spin up a process for each plane in the batch.
    This function runs in the process.

    For every new batch, main process sends a buffer token and plane index
    to this function. We process that plane and let the main process know
    we are done.
    """
    # more than about 4 threads seems to slow down computation
    torch.set_num_threads(min(n_threads, 4))

    while True:
        msg = process.get_msg_from_mainthread()
        if msg == EOFSignal:
            return
        # with torch multiprocessing, tensors are shared in memory - so
        # just update in place
        token, i = msg
        tensor, masks, enhanced_buffer = buffers[token]

        # plane gets edited in place
        plane = tensor[i : i + 1, :, :]
        plane_filter.clip_input(plane)
        masks[i : i + 1, :, :] = plane_filter.get_inside_mask(plane)
        enhanced_buffer[i : i + 1, :, :] = plane_filter.smooth_planes(plane)

        # tell the main thread we processed all the planes for this tensor
        process.send_msg_to_mainthread(None)


class VolumeFilter:
    """
    Filters and detects cells in the input data.

    This will take a 3d data array, filter each plane first with 2d filters
    finding bright spots. Then it filters the stack with a ball filter to
    find voxels that are potential cells. Then it runs cell detection on it
    to actually identify the cells.

    Parameters
    ----------
    settings : DetectionSettings
        Settings object that contains all the configuration data.
    """

    threshold_filter: ThresholdFilter3D | None = None

    def __init__(self, settings: DetectionSettings):
        self.settings = settings

        self.lap_filter = LaplacianFilter3D(
            voxel_sizes=settings.voxel_sizes,
            dtype=settings.filtering_dtype.__name__,
            batch_size=settings.batch_size,
            torch_device=settings.torch_device,
        )

        tile_size = settings.tiled_thresh_tile_size
        if tile_size:
            self.threshold_filter = ThresholdFilter3D(
                plane_height=settings.plane_height,
                plane_width=settings.plane_width,
                tile_xy_size=int(
                    round(tile_size * settings.soma_diameter_plane)
                ),
                tile_z_size=int(
                    round(tile_size * settings.soma_diameter_axial)
                ),
                n_sds_above_mean_thresh=self.settings.n_sds_above_mean_tiled_thresh,
                threshold_value=settings.threshold_value,
                dtype=settings.filtering_dtype.__name__,
                batch_size=settings.batch_size,
                torch_device=settings.torch_device,
            )

        self.ball_filter = BallFilter(
            plane_height=settings.plane_height,
            plane_width=settings.plane_width,
            ball_xy_size=settings.ball_xy_size,
            ball_z_size=settings.ball_z_size,
            overlap_fraction=settings.ball_overlap_fraction,
            threshold_value=settings.threshold_value,
            soma_centre_value=settings.soma_centre_value,
            tile_height=settings.tile_height,
            tile_width=settings.tile_width,
            dtype=settings.filtering_dtype.__name__,
            batch_size=settings.batch_size,
            torch_device=settings.torch_device,
            use_mask=True,
        )

        self.z = settings.start_plane

        self.cell_detector = CellDetector(
            settings.plane_height,
            settings.plane_width,
            start_z=self.z,
            soma_centre_value=settings.detection_soma_centre_value,
        )
        # make sure we load enough data to filter. Otherwise, we won't be ready
        # to filter and the data loading thread will wait for data to be
        # processed before loading more data, but that will never happen
        self.n_queue_buffer = max(
            self.settings.num_prefetch_batches,
            self.ball_filter.num_batches_before_ready,
        )
        self.n_queue_buffer = max(
            self.n_queue_buffer, self.lap_filter.num_batches_before_ready
        )

        if self.threshold_filter is not None:
            self.n_queue_buffer = max(
                self.n_queue_buffer,
                self.threshold_filter.num_batches_before_ready,
            )

    def _get_filter_buffers(
        self, cpu: bool, plane_filter: PlaneFilter
    ) -> List[Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        """
        Generates buffers to use for data loading and filtering.

        It creates pinned tensors ahead of time for faster copying to gpu.
        Pinned tensors are kept in RAM and are faster to copy to GPU because
        they can't be paged. So loaded data is copied to the tensor and then
        sent to the device.

        For CPU even though we don't pin, it's useful to create the buffers
        ahead of time and reuse it so we can filter in sub-processes
        (see `_plane_filter`).
        For tile masks, we only create buffers for CPU. On CUDA, they are
        generated every time new on the device.
        """
        batch_size = self.settings.batch_size
        torch_dtype = getattr(torch, self.settings.filtering_dtype.__name__)

        buffers = []
        for _ in range(self.n_queue_buffer):
            # the tensor used for data loading
            tensor = torch.empty(
                (batch_size, *self.settings.plane_shape),
                dtype=torch_dtype,
                pin_memory=not cpu and self.settings.pin_memory,
                device="cpu",
            )

            # tile mask and enhanced plane buffer - only for cpu because we use
            # second process to copy data into them. For gpu we get the buffers
            # directly from the functions that create then
            masks = None
            enhanced_planes = None
            if cpu:
                masks = plane_filter.get_tiled_buffer(
                    batch_size, self.settings.torch_device
                )
                enhanced_planes = torch.empty_like(tensor)

            buffers.append((tensor, masks, enhanced_planes))

        return buffers

    @inference_wrapper
    def _feed_signal_batches(
        self,
        thread: ThreadWithException,
        data: types.array,
        processors: List[ProcessWithException],
        buffers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        Runs in its own thread. It loads the input data planes, converts them
        to torch tensors of the right data-type, and sends them to cuda or to
        subprocesses for cpu to be filtered etc.
        """
        batch_size = self.settings.batch_size
        device = self.settings.torch_device
        start_plane = self.settings.start_plane
        end_plane = start_plane + self.settings.n_planes
        data_converter = self.settings.filter_data_converter_func
        cpu = self.settings.torch_device == "cpu"
        # should only have 2d filter processors on the cpu
        assert bool(processors) == cpu

        # seed our queue with tokens that give access to their buffers
        for token in range(len(buffers)):
            thread.send_msg_to_thread(token)

        for z in range(start_plane, end_plane, batch_size):
            # convert the input data to the right type
            np_data = data_converter(data[z : z + batch_size, :, :])
            # if we ran out of batches, we are done!
            n = np_data.shape[0]
            assert n

            # thread/underlying queues get first crack at msg. Unless we get
            # eof, this will block until a buffer is returned from the main
            # thread for reuse, or give us a buffer if we have unused buffers
            token = thread.get_msg_from_mainthread()
            if token is EOFSignal:
                return

            # buffer is free, get it from token
            tensor, masks, enhanced = buffers[token]

            # for last batch, it can be smaller than normal so only set up to n
            tensor[:n, :, :] = torch.from_numpy(np_data)
            tensor = tensor[:n, :, :]
            # on cuda, they are both None
            if masks is not None:
                masks = masks[:n]
            if enhanced is not None:
                enhanced = enhanced[:n]
            # For CPU, we send the tensor token to the 2d filtering processes
            # who have access to the tensors already. They overwrite the data
            # in the tensor with the filtered data and let the main thread
            # know. For GPU, we don't use these external processes, so we send
            # the tensor directly, after moving it to the device. Either way,
            # we don't reuse the tensor until the main thread let us know its
            # free
            if not cpu:
                # send to device - it won't block here if we pinned memory
                tensor = tensor.to(device=device, non_blocking=True)

            # if used, send each plane in batch to processor
            used_processors = []
            if cpu:
                used_processors = processors[:n]
                for i, process in enumerate(used_processors):
                    process.send_msg_to_thread((token, i))

            # tell the main thread to wait for processors (if used)
            msg = token, np_data, tensor, masks, enhanced, used_processors, n

            if n < batch_size:
                # on last batch, we are also done after this
                thread.send_msg_to_mainthread(msg)
                thread.send_msg_to_mainthread(_done_signal)
                return
            # send the data to the main thread
            thread.send_msg_to_mainthread(msg)

        thread.send_msg_to_mainthread(_done_signal)

    def process(
        self,
        plane_filter: PlaneFilter,
        signal_array,
        *,
        callback: Optional[Callable[[int], None]],
    ) -> None:
        """
        Takes the processor and the data and passes them through the filtering
        and cell detection stages.

        If the callback is provided, we call it after every plane with the
        current z index to update the status. It may be called from secondary
        threads.
        """
        progress_bar = tqdm(
            total=self.settings.n_planes, desc="Processing planes"
        )
        cpu = self.settings.torch_device == "cpu"
        n_threads = self.settings.n_torch_comp_threads

        # we re-use these tensors for data loading, so we have a fixed number
        # of planes in memory. The feeder thread will wait to load more data
        # until a tensor is free to be reused.
        # We have to keep the tensors in memory in main process while it's
        # in used elsewhere
        buffers = self._get_filter_buffers(cpu, plane_filter)

        # on cpu these processes will 2d filter each plane in the batch
        plane_processes = []
        if cpu:
            for _ in range(self.settings.batch_size):
                process = ProcessWithException(
                    target=_plane_filter,
                    args=(plane_filter, n_threads, buffers),
                    pass_self=True,
                )
                process.start()
                plane_processes.append(process)

        # thread that reads and sends us data
        feed_thread = ThreadWithException(
            target=self._feed_signal_batches,
            args=(signal_array, plane_processes, buffers),
            pass_self=True,
        )
        feed_thread.start()

        # thread that takes the 3d filtered data and does cell detection
        cells_thread = ThreadWithException(
            target=self._run_cell_detection_thread,
            args=(callback, progress_bar),
            pass_self=True,
        )
        cells_thread.start()

        try:
            self._process(feed_thread, cells_thread, plane_filter, cpu)
        finally:
            # if we end, make sure to tell the threads to stop
            feed_thread.notify_to_end_thread()
            cells_thread.notify_to_end_thread()
            for process in plane_processes:
                process.notify_to_end_thread()

            # the notification above ensures this won't block forever
            feed_thread.join()
            cells_thread.join()
            for process in plane_processes:
                process.join()

            # in case these threads sent us an exception but we didn't yet read
            # it, make sure to process them
            feed_thread.clear_remaining()
            cells_thread.clear_remaining()
            for process in plane_processes:
                process.clear_remaining()

        progress_bar.close()
        logger.debug("3D filter done")

    def _process(
        self,
        feed_thread: ThreadWithException,
        cells_thread: ThreadWithException,
        plane_filter: PlaneFilter,
        cpu: bool,
    ) -> None:
        """
        Processes the loaded data from feeder thread. If on cpu it is already
        2d filtered so just 3d filter. On cuda we need to do both 2d and 3d
        filtering. Then, it sends the filtered data off to the detection thread
        for cell detection.
        """
        processing_tokens = []
        lf = self.lap_filter
        tf = self.threshold_filter
        bf = self.ball_filter

        while True:
            # thread/underlying queues get first crack at msg. Unless we get
            # eof, this will block until we get more loaded data until no more
            # data or exception
            msg = feed_thread.get_msg_from_thread()
            # feeder thread exits at the end, causing an eof to be sent
            if msg is EOFSignal:
                break

            # if feeder thread has no more data, process remaining data already
            # waiting in tf/bf
            if msg == _done_signal:
                self._flush_filters(
                    feed_thread, cells_thread, processing_tokens, plane_filter
                )
                return

            # normal operation, msg is new 2d gaussian filtered data
            raw_data, masks, enhanced = self._unpack_feeder_thread_msg(
                msg, processing_tokens, cpu, plane_filter
            )

            # apply laplacian to enhance peaks
            lf.append(enhanced, raw_data, masks)
            if not lf.ready:
                continue
            lf.walk()
            enhanced = lf.get_processed_planes()
            raw_data, masks = lf.get_processed_side_data_planes()

            # threshold enhanced peaks using 2d per plane threshold
            binarized = [
                plane_filter.threshold_peak_enhanced_planes(p[None, ...])[
                    0, ...
                ]
                for p in enhanced
            ]

            # further threshold using tiles if using
            if tf is not None:
                # pass through tf if using
                tf.append(
                    enhanced,
                    binarized,
                    raw_data,
                    masks,
                )
                if not tf.ready:
                    # wait for more data before we can use any tf results
                    continue

                # it's ready to pass on to bf
                tf.walk()
                binarized = tf.get_processed_planes()
                raw_data, masks = tf.get_processed_side_data_planes()

            # apply ball filtering
            bf.append(binarized, masks, raw_data)
            # process bf data if ready and exit if detector asked
            if not self._send_ball_filter_result_to_cell_detector(
                feed_thread, cells_thread, processing_tokens
            ):
                return

    def _flush_filters(
        self,
        feed_thread: ThreadWithException,
        cells_thread: ThreadWithException,
        processing_tokens: list,
        plane_filter: PlaneFilter,
    ):
        lf = self.lap_filter
        bf = self.ball_filter
        tf = self.threshold_filter

        if lf.flush():
            assert lf.ready
            lf.walk()
            enhanced = lf.get_processed_planes()
            raw_data, masks = lf.get_processed_side_data_planes()

            binarized = [
                plane_filter.threshold_peak_enhanced_planes(p[None, ...])[
                    0, ...
                ]
                for p in enhanced
            ]

            # further threshold using tiles if using
            if tf is not None:
                # pass through tf if using
                tf.append(
                    enhanced,
                    binarized,
                    raw_data,
                    masks,
                )
                if tf.ready:
                    # it's ready to pass on to bf
                    tf.walk()
                    binarized = tf.get_processed_planes()
                    raw_data, masks = tf.get_processed_side_data_planes()

                    # apply ball filtering
                    bf.append(binarized, masks, raw_data)
                    # process bf data if ready and exit if detector asked
                    if not self._send_ball_filter_result_to_cell_detector(
                        feed_thread, cells_thread, processing_tokens
                    ):
                        return
            else:
                bf.append(binarized, masks, raw_data)
                # process bf data if ready and exit if detector asked
                if not self._send_ball_filter_result_to_cell_detector(
                    feed_thread, cells_thread, processing_tokens
                ):
                    return

        # check tf has no waiting data by flushing it, if there weren't
        # any new tf data, process remaining bf data and exit
        if tf is None or not tf.flush():
            if bf.flush():
                assert bf.ready
                self._send_ball_filter_result_to_cell_detector(
                    feed_thread, cells_thread, processing_tokens
                )
            return

        # else, need to finish processing flushed tf data first. If
        # flush returned true, no need to check if ready
        assert tf.ready
        tf.walk()
        planes = tf.get_processed_planes()
        raw_data, masks = tf.get_processed_side_data_planes()
        bf.append(planes, masks, raw_data)
        # first finish (newly) ready bf data, return if detector asks
        if not self._send_ball_filter_result_to_cell_detector(
            feed_thread, cells_thread, processing_tokens
        ):
            return

        # flush bf and process flushed data and exit
        if bf.flush():
            assert bf.ready
            self._send_ball_filter_result_to_cell_detector(
                feed_thread, cells_thread, processing_tokens
            )

    def _unpack_feeder_thread_msg(
        self,
        msg: tuple[
            int,
            np.ndarray,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            List[ProcessWithException],
            int,
        ],
        processing_tokens: list[int],
        cpu: bool,
        plane_filter: PlaneFilter,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        token, np_data, tensor, masks, enhanced, used_processors, n = msg
        # this token is in use until we return it
        processing_tokens.append(token)

        if cpu:
            # we did 2d filtering in different process. Make sure all
            # the planes are done filtering. Each msg from feeder
            # thread has corresponding msg for each used processor
            # (unless exception)
            for process in used_processors:
                process.get_msg_from_thread()
            # batch size can change at the end so resize buffer
            # todo: fix tracking of buffers so we don't have to clone
            masks = masks[:n, :, :].clone()
            enhanced = enhanced[:n, :, :].clone()
        else:
            # we're not doing 2d filtering in different process
            # tensor is edited in-place
            plane_filter.clip_input(tensor)
            masks = plane_filter.get_inside_mask(tensor)
            enhanced = plane_filter.smooth_planes(tensor)

        return np_data, masks, enhanced

    def _send_ball_filter_result_to_cell_detector(
        self,
        feed_thread: ThreadWithException,
        cells_thread: ThreadWithException,
        processing_tokens: list[int],
    ) -> bool:
        if not self.ball_filter.ready:
            return True

        self.ball_filter.walk()
        middle_planes = self.ball_filter.get_processed_planes()
        raw_planes = self.ball_filter.get_raw_planes()

        # at this point we know input tensor can be reused - return
        # it so feeder thread can load more data into it
        for token in processing_tokens:
            feed_thread.send_msg_to_thread(token)
        processing_tokens.clear()

        # thread/underlying queues get first crack at msg. Unless
        # we get eof, this will block until we get a token,
        # indicating we can send more data. The cells thread has a
        # fixed token supply, ensuring we don't send it too much
        # data, in case detection takes longer than filtering
        # Also, error messages incoming are at most # tokens behind
        token = cells_thread.get_msg_from_thread()
        if token is EOFSignal:
            return False
        # send it more data and return the token
        cells_thread.send_msg_to_thread((middle_planes, raw_planes, token))

        return True

    @inference_wrapper
    def _run_cell_detection_thread(
        self, thread: ThreadWithException, callback, progress_bar
    ) -> None:
        """
        Runs in its own thread and takes the filtered planes and passes them
        through the cell detection system. Also saves the planes as needed.
        """
        detector = self.cell_detector
        original_dtype = self.settings.plane_original_np_dtype
        detection_converter = self.settings.detection_data_converter_func
        save_planes = self.settings.save_planes
        previous_plane = None

        # main thread needs a token to send us planes - populate with some
        for _ in range(self.n_queue_buffer):
            thread.send_msg_to_mainthread(object())

        while True:
            # thread/underlying queues get first crack at msg. Unless we get
            # eof, this will block until we get more data
            msg = thread.get_msg_from_mainthread()
            # requested that we return. This can mean the main thread finished
            # sending data and it appended eof - so we get eof after all planes
            if msg is EOFSignal:
                return

            middle_planes, raw_planes, token = msg

            logger.debug(f"🏫 Detecting structures for planes {self.z}+")
            for raw_plane, plane in zip(raw_planes, middle_planes):
                if save_planes:
                    self.save_plane(plane.astype(original_dtype))

                # convert plane to the type needed by detection system
                # we should not need scaling because throughout
                # filtering we make sure result fits in this data type
                previous_plane = detector.process(
                    detection_converter(plane), previous_plane, raw_plane
                )

                if callback is not None:
                    callback(self.z)
                self.z += 1
                progress_bar.update()

            # we must return the token, otherwise the main thread will run out
            # and won't send more data to us
            thread.send_msg_to_mainthread(token)
            logger.debug(f"🏫 Structures done for planes {self.z}+")

    def save_plane(self, plane: np.ndarray) -> None:
        """
        Saves the plane as an image according to the settings.
        """
        if self.settings.plane_directory is None:
            raise ValueError(
                "plane_directory must be set to save planes to file"
            )
        # self.z is zero based, we should save names as 1-based.
        plane_name = self.settings.plane_prefix.format(n=self.z + 1) + ".tif"
        f_path = os.path.join(self.settings.plane_directory, plane_name)
        tifffile.imwrite(f_path, plane)

    def get_results(self, settings: DetectionSettings) -> List[Cell]:
        """
        Returns the detected cells.

        After filtering, this parses the resulting cells and splits large
        bright regions into individual cells.
        """
        logger.info("Splitting cell clusters and writing results")

        root_settings = self.settings
        max_cell_volume = settings.max_cell_volume
        use_centre_of_intensity = settings.detect_centre_of_intensity

        # valid cells
        cells = []
        # regions that must be split into cells
        needs_split = []
        structures = self.cell_detector.get_structures().items()
        intensities = self.cell_detector.get_structures_intensities()
        logger.debug(f"Processing {len(structures)} found cells")

        # first get all the cells that are not clusters
        for cell_id, cell_points in structures:
            intensity = None
            # if we don't use COI, don't pass on intensity. That'll disable it
            if use_centre_of_intensity:
                intensity = intensities[cell_id]
            cell_volume = len(cell_points)

            if cell_volume < max_cell_volume:
                cell_centre = get_structure_centre(cell_points, intensity)
                cells.append(Cell(cell_centre.tolist(), Cell.UNKNOWN))
            else:
                if cell_volume < settings.max_cluster_size:
                    needs_split.append((cell_id, cell_points, intensity))
                else:
                    cell_centre = get_structure_centre(cell_points, intensity)
                    cells.append(Cell(cell_centre.tolist(), Cell.ARTIFACT))

        if not needs_split:
            logger.debug("Finished splitting cell clusters - none found")
            return cells

        # now split clusters into cells
        logger.debug(f"Splitting {len(needs_split)} clusters")
        progress_bar = tqdm(
            total=len(needs_split), desc="Splitting cell clusters"
        )

        # the settings is pickled and re-created for each process, which is
        # important because splitting can modify the settings, so we don't want
        # parallel modifications for same object
        f = partial(_split_cells, settings=settings)
        ctx = mp.get_context("spawn")
        # we can't use the context manager because of coverage issues:
        # https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
        pool = ctx.Pool(processes=root_settings.n_processes)
        try:
            for cell_centres in pool.imap_unordered(f, needs_split):
                for cell_centre in cell_centres:
                    cells.append(Cell(cell_centre.tolist(), Cell.UNKNOWN))
                progress_bar.update()
        finally:
            pool.close()
            pool.join()

        progress_bar.close()
        logger.debug(
            f"Finished splitting cell clusters. Found {len(cells)} total cells"
        )

        return cells


@inference_wrapper
def _split_cells(arg, settings: DetectionSettings):
    # runs in its own process for a bright region to be split.
    # For splitting cells, we only run with one thread. Because the volume is
    # likely small and using multiple threads would cost more in overhead than
    # is worth. num threads can be set only at processes level.
    torch.set_num_threads(1)
    cell_id, cell_points, intensity = arg
    try:
        return split_cells(
            cell_points, settings=settings, intensity=intensity
        )[0]
    except (ValueError, AssertionError) as err:
        raise StructureSplitException(f"Cell {cell_id}, error; {err}")

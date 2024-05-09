import math
import multiprocessing as mp
import os
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import torch
from brainglobe_utils.cells.cells import Cell
from tifffile import tifffile
from tqdm import tqdm

from cellfinder.core import logger, types
from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter_cuda import BallFilter
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)
from cellfinder.core.tools.threading import EOFSignal, ThreadWithException
from cellfinder.core.tools.tools import inference_wrapper


class VolumeFilter:
    """
    Filters and detects cells in the input data.

    This will take a 3d data array, filter each plane first with 2d filters
    finding bright spots. Then it filters the stack with a ball filter to
    find voxels that are potentional cells. Then it runs cell detection on it
    to actually identify the cells.

    Parameters
    ----------
    settings : DetectionSettings
        Settings object that contains all the configuration data.
    """

    def __init__(self, settings: DetectionSettings):
        self.settings = settings

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
            dtype=settings.filterting_dtype,
            batch_size=settings.batch_size,
            torch_device=settings.torch_device,
            use_mask=True,
        )

        # todo: first z should account for middle plane not being start plane
        self.z = settings.start_plane + self.ball_filter.first_valid_plane

        self.cell_detector = CellDetector(
            settings.plane_height,
            settings.plane_width,
            start_z=self.z,
            soma_centre_value=settings.soma_centre_value,
        )
        # make sure we load enough data to filter. Otherwise, we won't be ready
        # to filter and the data loading thread will wait for data to be
        # processed before loading more data, but that will never happen
        self.n_queue_buffer = max(
            self.settings.num_prefetch_batches,
            self.ball_filter.num_batches_before_ready,
        )

    @inference_wrapper
    def _feed_signal_batches(self, data: types.array) -> None:
        """
        Runs in its own thread. It loads the input data planes, converts them
        to torch tensors of the right data-type, and sends them to the main
        thread to be filtered etc.
        """
        batch_size = self.settings.batch_size
        plane_shape = self.settings.plane_shape
        start_plane = self.settings.start_plane
        end_plane = start_plane + self.settings.n_planes
        data_converter = self.settings.filter_data_converter_func
        torch_dtype = getattr(torch, self.settings.filterting_dtype)

        thread = self.data_feed_thread

        # create pinned tensors ahead of time for faster copying to gpu.
        # Pinned tensors are kept in RAM and are faster to copy to GPU because
        # they can't be paged.
        # we re-use these tensors for data loading, so we have a fixed number
        # of planes in memory. This thread will wait to load more data until
        # a tensor is free to be reused
        for _ in range(self.n_queue_buffer):
            empty = torch.empty(
                (batch_size, *plane_shape), dtype=torch_dtype, pin_memory=True
            )
            # these tensors are free so send it to ourself
            thread.send_msg_to_thread(empty)

        for z in range(start_plane, end_plane, batch_size):
            # convert the data to the right type
            np_data = data_converter(data[z : z + batch_size, :, :])
            # if we ran out batches, we are done!
            n = np_data.shape[0]
            if not n:
                return

            # thread/underlying queues get first crack at msg. Unless we get
            # eof, this will block until a tensor is returned from the main
            # thread for reuse
            tensor = thread.get_msg_from_mainthread()
            if tensor is EOFSignal:
                return

            # for last batch, it can be smaller than normal so only set upto n
            tensor[:n, :, :] = torch.from_numpy(np_data)

            if n < batch_size:
                # or last batch, we are also done after this
                thread.send_msg_to_mainthread(tensor[:n, :, :])
                # we're done
                return
            # send the data to the main thread
            thread.send_msg_to_mainthread(tensor)

    def process(
        self,
        tile_processor: TileProcessor,
        signal_array,
        *,
        callback: Optional[Callable[[int], None]],
    ) -> None:
        """
        Takes the processor and the data and passes them through the filtering
        and cell detection stages.

        If the callback is provided, we call it after every batch with the
        current z index to update the status.
        """
        progress_bar = tqdm(
            total=self.settings.n_planes, desc="Processing planes"
        )
        # thread that loads and sends us data
        feed_thread = ThreadWithException(
            target=self._feed_signal_batches, args=(signal_array,)
        )
        self.data_feed_thread = feed_thread
        feed_thread.start()
        # thread that takes the filtered data and does cell detection
        cells_thread = ThreadWithException(
            target=self._run_filter_thread, args=(callback, progress_bar)
        )
        self.cells_thread = cells_thread
        cells_thread.start()

        try:
            while True:
                # thread/underlying queues get first crack at msg. Unless we
                # get eof, this will block until we get more loaded data
                tensor = feed_thread.get_msg_from_thread()
                if tensor is EOFSignal:
                    break

                # send to device - it won't block here because we pinned memory
                tensor = tensor.to(
                    device=self.settings.torch_device, non_blocking=True
                )
                # filter filter filter
                planes, masks = tile_processor.get_tile_mask(tensor)
                self.ball_filter.append(planes, masks)
                if self.ball_filter.ready:
                    self.ball_filter.walk()
                    middle_planes = self.ball_filter.get_processed_planes()

                    # at this point we know input tensor can be reused - return
                    # it so feeder thread can load more data into it
                    feed_thread.send_msg_to_thread(tensor)

                    # thread/underlying queues get first crack at msg. Unless
                    # we get eof, this will block until we get a token,
                    # indicating we can send more data. The cells thread has a
                    # fixed token supply, ensuring we don't send it too much
                    # data, in case detection takes longer than filtering
                    # (but typically that's not the slow part)
                    token = cells_thread.get_msg_from_thread()
                    if token is EOFSignal:
                        break
                    # send it more data and return the token
                    cells_thread.send_msg_to_thread((middle_planes, token))

        finally:
            # if we end, make sure to tell the threads to stop
            feed_thread.notify_to_end_thread()
            cells_thread.notify_to_end_thread()

        # the notification above ensures this won't block forever
        feed_thread.join()
        cells_thread.join()

        progress_bar.close()
        logger.debug("3D filter done")

    @inference_wrapper
    def _run_filter_thread(self, callback, progress_bar) -> None:
        """
        Runs in its own thread and takes the filtered planes and passes them
        through the cell detection system. Also saves the planes as needed.
        """
        thread = self.cells_thread
        detector = self.cell_detector
        detection_dtype = self.settings.detection_dtype
        previous_plane = None

        # main thread needs a token to send us planes - populate with some
        for _ in range(self.n_queue_buffer):
            thread.send_msg_to_mainthread(object())

        while True:
            # thread/underlying queues get first crack at msg. Unless we get
            # eof, this will block until we get more data
            msg = thread.get_msg_from_mainthread()
            # requested that we return
            if msg is EOFSignal:
                return

            # convert plane to the type needed by detection system
            middle_planes, token = msg
            middle_planes = middle_planes.astype(detection_dtype)

            logger.debug(f"🏐 Ball filtering planes {self.z}")
            if self.settings.save_planes:
                for plane in middle_planes:
                    self.save_plane(plane)

            logger.debug(f"🏫 Detecting structures for planes {self.z}")
            for plane in middle_planes:
                previous_plane = detector.process(plane, previous_plane)

                if callback is not None:
                    callback(self.z)
                self.z += 1
                progress_bar.update()

            # we must return the token, otherwise the main thread will run out
            # and won't send more data to us
            thread.send_msg_to_mainthread(token)
            logger.debug(f"🏫 Structures done for planes {self.z}")

    def save_plane(self, plane: np.ndarray) -> None:
        """
        Saves the plane as an image according to the settings.
        """
        if self.settings.plane_directory is None:
            raise ValueError(
                "plane_directory must be set to save planes to file"
            )
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.settings.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)

    def get_results(self, settings: DetectionSettings) -> List[Cell]:
        """
        Returns the detected cells.

        After filtering, this parses the resulting cells and splits large
        bright regions into individual cells.
        """
        logger.info("Splitting cell clusters and writing results")

        root_settings = self.settings
        max_cell_volume = sphere_volume(
            root_settings.soma_spread_factor * root_settings.soma_diameter / 2
        )

        # valid cells
        cells = []
        # regions that must be split into cells
        needs_split = []
        structures = self.cell_detector.get_structures().items()
        logger.debug(f"Processing {len(structures)} found cells")

        # first get all the cells that are not clusters
        for cell_id, cell_points in structures:
            cell_volume = len(cell_points)

            if cell_volume < max_cell_volume:
                cell_centre = get_structure_centre(cell_points)
                cells.append(Cell(cell_centre.tolist(), Cell.UNKNOWN))
            else:
                if cell_volume < settings.max_cluster_size:
                    needs_split.append((cell_id, cell_points))
                else:
                    cell_centre = get_structure_centre(cell_points)
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
        with ctx.Pool(processes=root_settings.n_processes) as pool:
            for cell_centres in pool.imap_unordered(f, needs_split):
                for cell_centre in cell_centres:
                    cells.append(Cell(cell_centre.tolist(), Cell.UNKNOWN))
                progress_bar.update()

        progress_bar.close()
        logger.debug(
            f"Finished splitting cell clusters. Found {len(cells)} total cells"
        )

        return cells


@inference_wrapper
def _split_cells(arg, settings: DetectionSettings):
    # runs in its own process for a bright region to be split
    # for splitting cells, we only run with one thread. Because the volume is
    # likely small and using multiple threads would cost more in overhead than
    # is worth
    torch.set_num_threads(1)
    cell_id, cell_points = arg
    try:
        return split_cells(cell_points, settings=settings)
    except (ValueError, AssertionError) as err:
        raise StructureSplitException(f"Cell {cell_id}, error; {err}")


def sphere_volume(radius: float) -> float:
    return (4 / 3) * math.pi * radius**3

import dataclasses
import math
import multiprocessing as mp
import os
from functools import partial, wraps
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


def inference_wrapper(func):
    @wraps(func)
    def inner_function(*args, **kwargs):
        with torch.inference_mode(True):
            return func(*args, **kwargs)

    return inner_function


class VolumeFilter(object):
    def __init__(self, settings: DetectionSettings):
        self.settings = settings

        self.previous_plane: Optional[np.ndarray] = None

        self.ball_filter = BallFilter(settings=settings)

        # todo: first z should account for middle plane not being start plane
        self.z = settings.start_plane + self.ball_filter.first_valid_plane

        self.cell_detector = CellDetector(
            width=settings.plane_dim1,
            height=settings.plane_dim2,
            start_z=self.z,
            soma_centre_value=settings.soma_centre_value,
        )
        self.n_queue_buffer = max(
            self.settings.num_prefetch_batches,
            self.ball_filter.num_batches_before_ready,
        )

    @inference_wrapper
    def _feed_signal_batches(self, data: types.array) -> None:
        batch_size = self.settings.batch_size
        plane_shape = self.settings.plane_shape
        start_plane = self.settings.start_plane
        end_plane = start_plane + self.settings.n_planes
        data_converter = self.settings.filter_data_converter_func
        torch_dtype = getattr(torch, self.settings.filterting_dtype)

        thread = self.data_feed_thread

        # create pinned tensors ahead of time for faster copying to gpu
        for _ in range(self.n_queue_buffer):
            empty = torch.empty(
                (batch_size, *plane_shape), dtype=torch_dtype, pin_memory=True
            )
            thread.send_msg_to_thread(empty)

        for z in range(start_plane, end_plane, batch_size):
            np_data = data_converter(data[z : z + batch_size, :, :])
            n = np_data.shape[0]
            if not n:
                return

            tensor = thread.get_msg_from_mainthread()
            if tensor is EOFSignal:
                return

            tensor[:n, :, :] = torch.from_numpy(np_data)

            if n < batch_size:
                thread.send_msg_to_mainthread(tensor[:n, :, :])
                # we're done
                return
            thread.send_msg_to_mainthread(tensor)

    def process(
        self,
        tile_processor: TileProcessor,
        signal_array,
        *,
        callback: Callable[[int], None],
    ) -> None:
        progress_bar = tqdm(
            total=self.settings.n_planes, desc="Processing planes"
        )

        feed_thread = ThreadWithException(
            target=self._feed_signal_batches, args=(signal_array,)
        )
        self.data_feed_thread = feed_thread
        feed_thread.start()

        cells_thread = ThreadWithException(
            target=self._run_filter_thread, args=(callback, progress_bar)
        )
        self.cells_thread = cells_thread
        cells_thread.start()

        try:
            while True:
                tensor = feed_thread.get_msg_from_thread()
                if tensor is EOFSignal:
                    break

                tensor = tensor.to(
                    device=self.settings.torch_device, non_blocking=True
                )
                planes, masks = tile_processor.get_tile_mask(tensor)
                self.ball_filter.append(planes, masks)
                if self.ball_filter.ready:
                    self.ball_filter.walk(True)
                    middle_planes = self.ball_filter.get_middle_planes()

                    # at this point we know input tensor can be reused - return
                    # it
                    feed_thread.send_msg_to_thread(tensor)

                    token = cells_thread.get_msg_from_thread()
                    if token is EOFSignal:
                        break
                    cells_thread.send_msg_to_thread((middle_planes, token))

        finally:
            feed_thread.notify_to_end_thread()
            cells_thread.notify_to_end_thread()

        feed_thread.join()
        cells_thread.join()

        progress_bar.close()
        logger.debug("3D filter done")

    @inference_wrapper
    def _run_filter_thread(self, callback, progress_bar) -> None:
        thread = self.cells_thread
        detector = self.cell_detector
        detection_dtype = self.settings.detection_dtype

        # main thread needs a token to send us planes - populate with some
        for _ in range(self.n_queue_buffer):
            thread.send_msg_to_mainthread(object())

        while True:
            msg = thread.get_msg_from_mainthread()
            if msg is EOFSignal:
                return

            middle_planes, token = msg
            middle_planes = middle_planes.astype(detection_dtype)

            logger.debug(f"🏐 Ball filtering plane {self.z}")
            if self.settings.save_planes:
                for plane in middle_planes:
                    self.save_plane(plane)

            logger.debug(f"🏫 Detecting structures for plane {self.z}")
            for plane in middle_planes:
                self.previous_plane = detector.process(
                    plane, self.previous_plane
                )

                if callback is not None:
                    callback(self.z)
                self.z += 1
                progress_bar.update()

            thread.send_msg_to_mainthread(token)
            logger.debug(f"🏫 Structures done for plane {self.z}")

    def save_plane(self, plane: np.ndarray) -> None:
        if self.settings.plane_directory is None:
            raise ValueError(
                "plane_directory must be set to save planes to file"
            )
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.settings.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)

    def get_results(self) -> List[Cell]:
        logger.info("Splitting cell clusters and writing results")

        settings = self.settings
        max_cell_volume = sphere_volume(
            settings.soma_spread_factor * settings.soma_diameter / 2
        )

        cells = []
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

        f = partial(_split_cells, settings_dict=dataclasses.asdict(settings))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=settings.n_processes) as pool:
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
def _split_cells(arg, settings_dict: dict):
    torch.set_num_threads(1)
    cell_id, cell_points = arg
    try:
        return split_cells(
            cell_points, settings=DetectionSettings(**settings_dict)
        )
    except (ValueError, AssertionError) as err:
        raise StructureSplitException(f"Cell {cell_id}, error; {err}")


def sphere_volume(radius: float) -> float:
    return (4 / 3) * math.pi * radius**3

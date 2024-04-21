import math
import multiprocessing.pool
import os
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from brainglobe_utils.cells.cells import Cell
from tifffile import tifffile
from tqdm import tqdm

from cellfinder.core import logger, types
from cellfinder.core.detect.filters.setup_filters import (
    get_ball_filter,
    get_cell_detector,
)
from cellfinder.core.detect.filters.volume.structure_detection import (
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)


class VolumeFilter(object):
    def __init__(
        self,
        *,
        soma_diameter: float,
        soma_size_spread_factor: float = 1.4,
        setup_params: Tuple[np.ndarray, Any, int, int, float, Any],
        n_planes: int,
        n_locks_release: int,
        save_planes: bool = False,
        plane_directory: Optional[str] = None,
        start_plane: int = 0,
        max_cluster_size: int = 5000,
        outlier_keep: bool = False,
        artifact_keep: bool = True,
        batch_size: int = 1,
        torch_device: str = "cpu",
    ):
        self.soma_diameter = soma_diameter
        self.soma_size_spread_factor = soma_size_spread_factor
        self.n_planes = n_planes
        # todo: first z should account for middle plane not being start plane
        self.z = start_plane
        self.save_planes = save_planes
        self.plane_directory = plane_directory
        self.max_cluster_size = max_cluster_size
        self.outlier_keep = outlier_keep
        self.n_locks_release = n_locks_release
        self.batch_size = batch_size
        self.torch_device = torch_device

        self.artifact_keep = artifact_keep

        self.clipping_val = None
        self.threshold_value = None
        self.setup_params = setup_params

        self.previous_plane: Optional[np.ndarray] = None

        self.ball_filter = get_ball_filter(
            plane=self.setup_params[0],
            soma_diameter=self.setup_params[1],
            ball_xy_size=self.setup_params[2],
            ball_z_size=self.setup_params[3],
            ball_overlap_fraction=self.setup_params[4],
            batch_size=batch_size,
            torch_dtype=torch.float32,
            torch_device=torch_device,
        )

        self.cell_detector = get_cell_detector(
            plane_shape=self.setup_params[0].shape,  # type: ignore
            ball_z_size=self.setup_params[3],
            z_offset=self.setup_params[5],
        )

        self.times = []
        self.n_buffered_batches = max(
            max(3, 1), self.ball_filter.num_batches_before_ready
        )

    def feed_signal_batches(
        self,
        incoming_tensors: Queue,
        outgoing_tensors: Queue,
        data: types.array,
    ) -> None:
        try:
            batch_size = self.batch_size
            plane_shape = self.setup_params[0].shape[::-1]

            for _ in range(self.n_buffered_batches):
                incoming_tensors.put(
                    (
                        "tensor",
                        torch.empty(
                            (batch_size, *plane_shape),
                            dtype=torch.float32,
                            pin_memory=True,
                        ),
                    )
                )

            for z in range(0, self.n_planes, batch_size):
                np_data = np.asarray(
                    data[z : z + batch_size, :, :], dtype=np.float32
                )
                np_data = np.moveaxis(np_data, 1, 2)
                if not np_data.shape[0]:
                    return

                msg, tensor = incoming_tensors.get(block=True, timeout=None)
                if msg == "eof":
                    return

                tensor[: np_data.shape[0], :, :] = torch.from_numpy(np_data)
                outgoing_tensors.put(("tensor", tensor))
        except Exception as e:
            outgoing_tensors.put(("exception", e))
        finally:
            outgoing_tensors.put(("eof", None))

    def process(
        self,
        get_tile_mask,
        signal_array,
        *,
        callback: Callable[[int], None],
    ) -> None:
        progress_bar = tqdm(total=self.n_planes, desc="Processing planes")

        data_queue_in = Queue(maxsize=0)
        data_queue_back = Queue(maxsize=0)
        data_thread = Thread(
            target=self.feed_signal_batches,
            args=(data_queue_back, data_queue_in, signal_array),
        )
        data_thread.start()

        filter_queue = Queue(maxsize=self.n_buffered_batches)
        filter_thread = Thread(
            target=self._run_filter_thread,
            args=(filter_queue, callback, progress_bar),
        )
        filter_thread.start()

        try:
            while True:
                msg, value = data_queue_in.get(block=True, timeout=None)
                if msg == "eof":
                    break
                if msg == "exception":
                    raise Exception("Processing signal data failed") from value
                tensor = value

                tensor = tensor.to(device=self.torch_device, non_blocking=True)
                planes, masks = get_tile_mask(tensor)
                self.ball_filter.append(planes, masks)
                if self.ball_filter.ready:
                    self.ball_filter.walk(True)
                    middle_planes = self.ball_filter.get_middle_planes()

                    # at this point we know input tensor can be reused - return
                    # it
                    data_queue_back.put(("tensor", tensor))
                    filter_queue.put(
                        ("tensor", middle_planes), block=True, timeout=None
                    )

        finally:
            data_queue_back.put(("eof", None))
            filter_queue.put(("eof", None))

        data_thread.join()
        filter_thread.join()
        progress_bar.close()
        logger.debug("3D filter done")
        # np.save("/home/matte/times2.npy", np.asarray(self.times))
        raise ValueError

    def _run_filter_thread(
        self, incoming_tensors: Queue, callback, progress_bar
    ) -> None:
        while True:
            msg, middle_planes = incoming_tensors.get(block=True, timeout=None)
            if msg == "eof":
                return

            middle_planes = middle_planes.astype(np.uint32)
            middle_planes[middle_planes >= 2**24 - 2] = np.iinfo(np.uint32).max

            logger.debug(f"🏐 Ball filtering plane {self.z}")
            # filtering original images, the images should be large enough in
            # x/y to benefit from parallelization. Note: don't pass arg as
            # keyword arg because numba gets stuck (probably b/c class jit is
            # new)
            if self.save_planes:
                for plane in middle_planes:
                    self.save_plane(plane)

            logger.debug(f"🏫 Detecting structures for plane {self.z}")
            for plane in middle_planes:
                self.previous_plane = self.cell_detector.process(
                    plane, self.previous_plane
                )

                callback(self.z)
                self.z += 1
                progress_bar.update()

            logger.debug(f"🏫 Structures done for plane {self.z}")

    def save_plane(self, plane: np.ndarray) -> None:
        if self.plane_directory is None:
            raise ValueError(
                "plane_directory must be set to save planes to file"
            )
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)

    def get_results(self, worker_pool: multiprocessing.Pool) -> List[Cell]:
        logger.info("Splitting cell clusters and writing results")

        max_cell_volume = sphere_volume(
            self.soma_size_spread_factor * self.soma_diameter / 2
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
                if cell_volume < self.max_cluster_size:
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

        # we are not returning Cell instances from func because it'd be pickled
        # by multiprocess which slows it down
        func = partial(_split_cells, outlier_keep=self.outlier_keep)
        for cell_centres in worker_pool.imap_unordered(func, needs_split):
            for cell_centre in cell_centres:
                cells.append(Cell(cell_centre.tolist(), Cell.UNKNOWN))
            progress_bar.update()

        progress_bar.close()
        logger.debug(
            f"Finished splitting cell clusters. Found {len(cells)} total cells"
        )

        return cells


def _split_cells(arg, outlier_keep):
    cell_id, cell_points = arg
    try:
        return split_cells(cell_points, outlier_keep=outlier_keep)
    except (ValueError, AssertionError) as err:
        raise StructureSplitException(f"Cell {cell_id}, error; {err}")


def sphere_volume(radius: float) -> float:
    return (4 / 3) * math.pi * radius**3

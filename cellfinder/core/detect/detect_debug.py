import dataclasses
import math
import multiprocessing as mp
import pickle
from enum import IntEnum, auto
from functools import cached_property, partial, wraps
from pathlib import Path
from queue import Empty
from typing import Callable, Optional, Sequence, Type

import numpy as np
import tifffile
import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.IO.cells import save_cells
from brainglobe_utils.IO.image.load import read_z_stack
from brainglobe_utils.IO.yaml import save_yaml

from cellfinder.core import types
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
from cellfinder.core.tools.threading import EOFSignal, ThreadWithException
from cellfinder.core.tools.tools import inference_wrapper


def thread_run(thread: ThreadWithException):
    while True:
        # if the main thread wants us to exit, it'll wake us up
        msg = thread.get_msg_from_mainthread()
        # we were asked to exit
        if msg == EOFSignal:
            return

        f, obj, args, kwargs = msg
        f(obj, *args, **kwargs)


def run_in_thread(func) -> Callable:
    @wraps(func)
    def inner(obj, *args, **kwargs):
        thread: ThreadWithException | None = getattr(
            obj, "thread_runner", None
        )
        if thread is None:
            func(obj, *args, **kwargs)
        else:
            thread.send_msg_to_thread((func, obj, args, kwargs))

    return inner


class DetectionStage(IntEnum):

    input = auto()
    clipped = auto()
    peak_enhanced = auto()
    bin_2d_peaks = auto()
    bin_3d_peaks = auto()
    filtered_3d_ball = auto()
    structs_detection = auto()
    struct_typed = auto()
    struct_splitting = auto()


class DataStore:

    settings: DetectionSettings

    input_data: types.array | None = None

    clipped_data: types.array | None = None

    inside_data: types.array | None = None

    peak_enhanced_data: types.array | None = None

    bin_2d_peaks_data: types.array | None = None

    bin_3d_peaks_data: types.array | None = None

    filtered_3d_ball_data: types.array | None = None

    current_batch: int = 0

    _input_batch: tuple[torch.Tensor, np.ndarray] | None = None

    _clipped_batch: torch.Tensor | None = None

    _inside_batch: torch.Tensor | None = None

    _peak_enhanced_batch: (
        tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]] | None
    ) = None

    _bin_2d_peaks_batch: (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[np.ndarray],
            list[torch.Tensor],
        ]
        | None
    ) = None

    _bin_3d_peaks_batch: (
        tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]] | None
    ) = None

    _filtered_3d_ball_batch: (
        tuple[list[np.ndarray], list[np.ndarray]] | None
    ) = None

    def __init__(
        self,
        detection: "DetectionDebug",
        local_store_loader: Optional["DetectionDebug"],
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
        start_plane: int = 0,
        end_plane: int = 0,
        bottom_corner: tuple[int, int] = (0, 0),
        top_corner: tuple[int, int] = (0, 0),
    ):
        self.settings = detection.settings
        self.detection = detection
        self.lsl = local_store_loader
        self.start_gen_from = start_gen_from
        self.end_gen_on = end_gen_on
        self.start_plane = start_plane
        self.end_plane = end_plane
        self.bottom_corner = bottom_corner
        self.top_corner = top_corner

    def crop_image_if_needed(self, data: types.array) -> types.array:
        bot_y, bot_x = self.bottom_corner
        top_y, top_x = self.top_corner

        start_plane = max(self.start_plane, 0)
        bot_y = max(bot_y, 0)
        bot_x = max(bot_x, 0)

        end_plane = self.end_plane
        if end_plane <= 0:
            end_plane = data.shape[0]
        if top_y <= 0:
            top_y = data.shape[1]
        if top_x <= 0:
            top_x = data.shape[2]

        return data[start_plane:end_plane, bot_y:top_y, bot_x:top_x]

    def _load_stack(self, prefix: str) -> types.array:
        lsl = self.lsl
        if lsl is None:
            data = self.detection.get_image(local_prefix=prefix)
        else:
            data = self.detection.get_image(
                arr_or_path=getattr(lsl, f"{prefix}_image_path")
            )
        return self.crop_image_if_needed(data)

    def set_current_batch(self, batch: int) -> None:
        self.current_batch = batch
        self._input_batch = None
        self._clipped_batch = None
        self._peak_enhanced_batch = None
        self._inside_batch = None
        self._bin_2d_peaks_batch = None
        self._bin_3d_peaks_batch = None
        self._filtered_3d_ball_batch = None

    @property
    def input_batch(self) -> tuple[torch.Tensor, np.ndarray]:
        if self._input_batch is not None:
            return self._input_batch

        if self.input_data is None:
            self.input_data = self._load_stack("input")

        batch_size = self.detection.batch_size
        i = self.current_batch

        batch_np = np.asarray(self.input_data[i : i + batch_size]).astype(
            self.settings.plane_original_np_dtype
        )
        batch_np = self.settings.filter_data_converter_func(batch_np)
        batch_torch = torch.from_numpy(batch_np).to(
            self.detection.torch_device
        )

        return batch_torch, batch_np

    @input_batch.setter
    def input_batch(self, value: tuple[torch.Tensor, np.ndarray]) -> None:
        self._input_batch = value

    @property
    def clipped_input_batch(self) -> torch.Tensor:
        if self._clipped_batch is not None:
            return self._clipped_batch

        if self.clipped_data is None:
            self.clipped_data = self._load_stack("clipped_input")

        batch_size = self.detection.batch_size
        i = self.current_batch

        batch_clipped = self.clipped_data[i : i + batch_size]
        batch_clipped = np.asarray(
            batch_clipped, dtype=self.settings.filtering_dtype
        )
        batch_clipped = torch.from_numpy(batch_clipped).to(
            self.detection.torch_device
        )
        return batch_clipped

    @clipped_input_batch.setter
    def clipped_input_batch(self, value: torch.Tensor) -> None:
        self._clipped_batch = value

    @property
    def inside_batch(self) -> torch.Tensor:
        if self._inside_batch is not None:
            return self._inside_batch

        if self.inside_data is None:
            self.inside_data = self._load_stack("inside_brain")

        batch_size = self.detection.batch_size
        i = self.current_batch

        inside_brain_tiles = self.inside_data[i : i + batch_size]
        inside_brain_tiles = np.asarray(inside_brain_tiles, dtype=bool)
        inside_brain_tiles = torch.from_numpy(inside_brain_tiles).to(
            self.detection.torch_device
        )
        return inside_brain_tiles

    @inside_batch.setter
    def inside_batch(self, value: torch.Tensor) -> None:
        self._inside_batch = value

    @property
    def peak_enhanced_batch(
        self,
    ) -> tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]]:
        if self._peak_enhanced_batch is not None:
            return self._peak_enhanced_batch

        if self.peak_enhanced_data is None:
            self.peak_enhanced_data = self._load_stack("peak_enhanced")

        batch_size = self.detection.batch_size
        i = self.current_batch

        enhanced_planes = self.peak_enhanced_data[i : i + batch_size]
        enhanced_planes = np.asarray(
            enhanced_planes, dtype=self.settings.filtering_dtype
        )
        enhanced_planes = torch.from_numpy(enhanced_planes).to(
            self.detection.torch_device
        )
        return (
            list(enhanced_planes),
            list(self.input_batch[1]),
            list(self.inside_batch),
        )

    @peak_enhanced_batch.setter
    def peak_enhanced_batch(
        self,
        value: tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]],
    ) -> None:
        self._peak_enhanced_batch = value

    @property
    def bin_2d_peaks_batch(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[np.ndarray],
        list[torch.Tensor],
    ]:
        if self._bin_2d_peaks_batch is not None:
            return self._bin_2d_peaks_batch

        if self.bin_2d_peaks_data is None:
            self.bin_2d_peaks_data = self._load_stack("bin_2d_peaks")

        batch_size = self.detection.batch_size
        i = self.current_batch

        bin_2d_peaks = self.bin_2d_peaks_data[i : i + batch_size]
        bin_2d_peaks = np.asarray(bin_2d_peaks)
        bin_2d_peaks = torch.from_numpy(bin_2d_peaks).to(
            self.detection.torch_device
        )
        enhanced, np_data, inside = self.peak_enhanced_batch

        return (
            enhanced,
            list(bin_2d_peaks),
            np_data,
            inside,
        )

    @bin_2d_peaks_batch.setter
    def bin_2d_peaks_batch(
        self,
        value: tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[np.ndarray],
            list[torch.Tensor],
        ],
    ) -> None:
        self._bin_2d_peaks_batch = value

    @property
    def bin_3d_peaks_batch(
        self,
    ) -> tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]]:
        if self._bin_3d_peaks_batch is not None:
            return self._bin_3d_peaks_batch

        if self.bin_3d_peaks_data is None:
            self.bin_3d_peaks_data = self._load_stack("bin_3d_peaks")

        batch_size = self.detection.batch_size
        i = self.current_batch

        bin_3d_peaks = self.bin_3d_peaks_data[i : i + batch_size]
        bin_3d_peaks = np.asarray(bin_3d_peaks)
        bin_3d_peaks = torch.from_numpy(bin_3d_peaks).to(
            self.detection.torch_device
        )
        return (
            list(bin_3d_peaks),
            list(self.input_batch[1]),
            list(self.inside_batch),
        )

    @bin_3d_peaks_batch.setter
    def bin_3d_peaks_batch(
        self,
        value: tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]],
    ) -> None:
        self._bin_3d_peaks_batch = value

    @property
    def filtered_3d_ball_batch(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self._filtered_3d_ball_batch is not None:
            return self._filtered_3d_ball_batch

        if self.filtered_3d_ball_data is None:
            self.filtered_3d_ball_data = self._load_stack("filtered_3d_ball")

        batch_size = self.detection.batch_size
        i = self.current_batch

        filtered_3d_ball = self.filtered_3d_ball_data[i : i + batch_size]
        filtered_3d_ball = np.asarray(filtered_3d_ball)
        return list(filtered_3d_ball), list(self.input_batch[1])

    @filtered_3d_ball_batch.setter
    def filtered_3d_ball_batch(
        self, value: tuple[list[np.ndarray], list[np.ndarray]]
    ) -> None:
        self._filtered_3d_ball_batch = value


class DetectionDebug:
    """Saves everything to the data store."""

    settings: DetectionSettings

    splitting_settings: DetectionSettings

    input_settings: dict

    data_store: DataStore | None = None

    thread_runner: ThreadWithException | None = None

    def __init__(
        self,
        signal_shape: tuple[int, int, int],  # expect to load z, y, x
        local_store: Path | str,
        batch_size: int = 1,
        torch_device="cpu",
        dtype=np.uint16,
        use_scipy=None,
        voxel_sizes: tuple[float, float, float] = (5, 2, 2),
        soma_diameter: float = 16,
        max_cluster_size: float = 100_000,
        ball_xy_size: float = 6,
        ball_z_size: float = 15,
        ball_overlap_fraction: float = 0.6,
        soma_spread_factor: float = 1.4,
        n_free_cpus: int = 2,
        log_sigma_size: float = 0.2,
        n_sds_above_mean_thresh: float = 10,
        detect_centre_of_intensity: bool = False,
        split_ball_xy_size: float = 6,
        split_ball_z_size: float = 15,
        split_ball_overlap_fraction: float = 0.8,
        n_splitting_iter: int = 10,
        n_sds_above_mean_tiled_thresh: float = 10,
        tiled_thresh_tile_size: float | None = None,
    ):
        self.signal_shape = signal_shape
        self.batch_size = batch_size
        self.torch_device = torch_device
        if use_scipy is None:
            use_scipy = torch_device != "cuda"
        self.use_scipy = use_scipy
        self.local_store = Path(local_store)
        self.input_settings = {}

        self.settings = DetectionSettings(
            plane_original_np_dtype=dtype,
            plane_shape=signal_shape[1:],
            voxel_sizes=voxel_sizes,
            soma_spread_factor=soma_spread_factor,
            soma_diameter_um=soma_diameter,
            max_cluster_size_um3=max_cluster_size,
            ball_xy_size_um=ball_xy_size,
            ball_z_size_um=ball_z_size,
            start_plane=0,
            end_plane=signal_shape[0],
            n_free_cpus=n_free_cpus,
            ball_overlap_fraction=ball_overlap_fraction,
            log_sigma_size=log_sigma_size,
            n_sds_above_mean_thresh=n_sds_above_mean_thresh,
            outlier_keep=False,
            artifact_keep=False,
            save_planes=False,
            batch_size=batch_size,
            torch_device=torch_device,
            n_splitting_iter=n_splitting_iter,
            detect_centre_of_intensity=detect_centre_of_intensity,
            tiled_thresh_tile_size=tiled_thresh_tile_size,
            n_sds_above_mean_tiled_thresh=n_sds_above_mean_tiled_thresh,
        )

        kwargs = dataclasses.asdict(self.settings)
        kwargs["ball_z_size_um"] = split_ball_z_size
        kwargs["ball_xy_size_um"] = split_ball_xy_size
        kwargs["ball_overlap_fraction"] = split_ball_overlap_fraction
        kwargs["torch_device"] = "cpu"
        kwargs["plane_original_np_dtype"] = np.float32
        self.splitting_settings = DetectionSettings(**kwargs)

    @property
    def structures_data_path(self) -> Path:
        return self.local_store / "structure_data.pickle"

    @property
    def initial_cell_candidates_path(self) -> Path:
        return self.local_store / "struct_cell_candidate.yml"

    @property
    def structs_needs_split_path(self) -> Path:
        return self.local_store / "struct_needs_split.yml"

    @property
    def structs_too_big_path(self) -> Path:
        return self.local_store / "struct_too_big.yml"

    @property
    def struct_split_into_cell_candidate_path(self) -> Path:
        return self.local_store / "struct_split_into_cell_candidate.yml"

    @property
    def input_image_path(self) -> Path:
        return self.local_store / "input"

    @property
    def clipped_input_image_path(self) -> Path:
        return self.local_store / "clipped_input"

    @property
    def peak_enhanced_image_path(self) -> Path:
        return self.local_store / "peak_enhanced"

    @property
    def inside_brain_image_path(self) -> Path:
        return self.local_store / "inside_brain"

    @property
    def bin_2d_peaks_image_path(self) -> Path:
        return self.local_store / "bin_2d_peaks"

    @property
    def bin_3d_peaks_image_path(self) -> Path:
        return self.local_store / "bin_3d_peaks"

    @property
    def filtered_3d_ball_image_path(self) -> Path:
        return self.local_store / "filtered_3d_ball"

    @property
    def structs_id_image_path(self) -> Path:
        return self.local_store / "structs_id"

    @property
    def struct_type_image_path(self) -> Path:
        return self.local_store / "struct_type"

    @property
    def struct_type_split_image_path(self) -> Path:
        return self.local_store / "struct_type_split"

    @property
    def config_yaml_path(self) -> Path:
        return self.local_store / "detection_config.yml"

    @cached_property
    def plane_filter(self) -> PlaneFilter:
        settings = self.settings
        return PlaneFilter(
            plane_shape=self.signal_shape[1:],
            clipping_value=settings.clipping_value,
            threshold_value=settings.threshold_value,
            n_sds_above_mean_thresh=settings.n_sds_above_mean_thresh,
            soma_diameter=settings.soma_diameter_plane,
            log_sigma_size=settings.log_sigma_size,
            torch_device=self.torch_device,
            dtype=settings.filtering_dtype.__name__,
            use_scipy=self.use_scipy,
        )

    @cached_property
    def lap_filter(self) -> LaplacianFilter3D | None:
        settings = self.settings
        return LaplacianFilter3D(
            voxel_sizes=settings.voxel_sizes,
            dtype=settings.filtering_dtype.__name__,
            batch_size=settings.batch_size,
            torch_device=settings.torch_device,
        )

    @cached_property
    def threshold_filter(self) -> ThresholdFilter3D | None:
        settings = self.settings
        tile_size = settings.tiled_thresh_tile_size
        if not tile_size:
            return None

        return ThresholdFilter3D(
            plane_height=settings.plane_height,
            plane_width=settings.plane_width,
            tile_xy_size=int(round(tile_size * settings.soma_diameter_plane)),
            tile_z_size=int(round(tile_size * settings.soma_diameter_axial)),
            n_sds_above_mean_thresh=self.settings.n_sds_above_mean_tiled_thresh,
            threshold_value=settings.threshold_value,
            dtype=settings.filtering_dtype.__name__,
            batch_size=settings.batch_size,
            torch_device=settings.torch_device,
        )

    @cached_property
    def ball_filter(self) -> BallFilter:
        settings = self.settings
        return BallFilter(
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
            batch_size=self.batch_size,
            torch_device=self.torch_device,
            use_mask=True,
        )

    @cached_property
    def cell_detector(self) -> CellDetector:
        settings = self.settings
        return CellDetector(
            settings.plane_height,
            settings.plane_width,
            0,
            soma_centre_value=settings.detection_soma_centre_value,
        )

    @classmethod
    def needs_crop(
        cls,
        bottom_corner: tuple[int, int] = (0, 0),
        top_corner: tuple[int, int] = (0, 0),
    ) -> bool:
        bot_y, bot_x = bottom_corner
        top_y, top_x = top_corner
        crop = (
            bot_y >= 0
            and bot_x >= 0
            and top_y > 0
            and top_x > 0
            and bot_y < top_y
            and bot_x < top_x
        )
        return crop

    @run_in_thread
    def save_tiffs(
        self,
        prefix: str,
        start_index: int,
        buffer: (
            torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray]
        ),
    ) -> None:
        root = self.local_store / prefix
        root.mkdir(parents=True, exist_ok=True)

        if isinstance(buffer, torch.Tensor):
            planes = list(buffer.cpu().numpy())
        elif isinstance(buffer, np.ndarray):
            planes = list(buffer)
        else:
            planes = []
            for p in buffer:
                if isinstance(p, torch.Tensor):
                    planes.append(p.cpu().numpy())
                else:
                    planes.append(p)

        digits = int(math.ceil(math.log10(self.signal_shape[0])))
        for i, plane in enumerate(planes, start_index):
            tifffile.imwrite(
                root / f"{prefix}_{i:0{digits}}.tif",
                plane,
                compression="LZW",
            )

    def get_image(
        self,
        arr_or_path: Path | str | types.array | None = None,
        local_prefix: str = "",
        dtype: Type | None = None,
    ) -> types.array:
        if arr_or_path is not None:
            data = arr_or_path
            if not isinstance(arr_or_path, types.array):
                data = read_z_stack(str(arr_or_path))
        elif local_prefix:
            data = read_z_stack(str(self.local_store / local_prefix))
        else:
            raise ValueError(
                "Either an array, image, path or local store prefix must "
                "be provided"
            )

        if dtype is not None:
            data = data.astype(dtype)

        return data

    def _load_detected_structs(
        self,
        local_store_loader: "DetectionDebug" = None,
        start_plane: int = 0,
        end_plane: int = 0,
        bottom_corner: tuple[int, int] = (0, 0),
        top_corner: tuple[int, int] = (0, 0),
    ) -> None:
        bot_y, bot_x = bottom_corner
        top_y, top_x = top_corner

        crop = self.needs_crop(bottom_corner, top_corner)
        needs_z_slice = end_plane > 0 and start_plane >= 0

        cell_detector = self.cell_detector

        structures_data_path = self.structures_data_path
        if local_store_loader is not None:
            structures_data_path = local_store_loader.structures_data_path
        with open(structures_data_path, "rb") as fh:
            structs = pickle.load(fh)

        for sid, arr, intensity in structs:
            # check first z, so we don't need to check those outside the z
            # planes during cropping
            if needs_z_slice:
                mask = (start_plane <= arr[:, 2]) & (arr[:, 2] < end_plane)

                if not np.any(mask):
                    continue
                arr = arr[mask, :]
                intensity = intensity[mask]

            if crop:
                mask = (bot_x <= arr[:, 0]) & (arr[:, 0] < top_x)
                mask = mask & (bot_y <= arr[:, 1]) & (arr[:, 1] < top_y)

                if not np.any(mask):
                    continue
                arr = arr[mask, :]
                intensity = intensity[mask]

            cell_detector.add_points(sid, arr, intensity)

    def load_data(
        self,
        signal: Path | str | np.ndarray | None = None,
        local_store_loader: "DetectionDebug" = None,
        start_gen_from: DetectionStage = DetectionStage.input,
        end_gen_on: DetectionStage = DetectionStage.struct_splitting,
        start_plane: int = 0,
        end_plane: int = 0,
        bottom_corner: tuple[int, int] = (0, 0),
        top_corner: tuple[int, int] = (0, 0),
    ) -> None:
        self.input_settings = {
            "start_gen_from": start_gen_from,
            "end_gen_on": end_gen_on,
            "start_plane": start_plane,
            "end_plane": end_plane,
            "bottom_corner": bottom_corner,
            "top_corner": top_corner,
            "input": (
                ""
                if signal is None or isinstance(signal, types.array)
                else str(signal)
            ),
        }
        data_store = DataStore(
            self,
            local_store_loader,
            start_gen_from,
            end_gen_on,
            start_plane,
            end_plane,
            bottom_corner,
            top_corner,
        )
        self.data_store = data_store

        # We load from original input only if we start at input because that's
        # the first input to pipeline
        if start_gen_from == DetectionStage.input <= end_gen_on:
            if signal is not None:
                data = self.get_image(arr_or_path=signal)
            elif local_store_loader is not None:
                data = self.get_image(
                    arr_or_path=local_store_loader.input_image_path
                )
            else:
                # can't get input data from our store, if we start with input
                raise ValueError("Input data not provided")

            data_store.input_data = data_store.crop_image_if_needed(data)

        # if starting from before splitting, we generate the cell data anew so
        # only load existing data if starting at splitting
        if start_gen_from == DetectionStage.struct_splitting <= end_gen_on:
            self._load_detected_structs(
                local_store_loader,
                start_plane,
                end_plane,
                bottom_corner,
                top_corner,
            )

    def batch_input(
        self, i, start_gen_from: DetectionStage, end_gen_on: DetectionStage
    ) -> None:
        if start_gen_from <= DetectionStage.input <= end_gen_on:
            batch_torch, batch_np = self.data_store.input_batch

            # we don't transform the input so just set it
            self.data_store.input_batch = batch_torch, batch_np
            self.save_tiffs("input", i, batch_np)

    def batch_clip(
        self,
        i,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ) -> None:
        if start_gen_from <= DetectionStage.clipped <= end_gen_on:
            batch_torch, _ = self.data_store.input_batch
            batch_clipped = torch.clone(batch_torch)

            self.plane_filter.clip_input(batch_clipped)
            inside_brain_tiles = self.plane_filter.get_inside_mask(
                batch_clipped
            )

            self.data_store.clipped_input_batch = batch_clipped
            self.data_store.inside_batch = inside_brain_tiles

            self.save_tiffs("clipped_input", i, batch_clipped.cpu().numpy())
            self.save_tiffs(
                "inside_brain", i, inside_brain_tiles.cpu().numpy()
            )

    def batch_peak_enhanced(
        self,
        i,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
        do_flush: bool = False,
    ) -> int | None:
        lf = self.lap_filter

        if not (start_gen_from <= DetectionStage.peak_enhanced <= end_gen_on):
            return None

        if do_flush:
            if not lf.flush():
                return i
        else:
            _, batch_np = self.data_store.input_batch
            smoothed = self.plane_filter.smooth_planes(
                self.data_store.clipped_input_batch
            )
            lf.append(
                smoothed,
                batch_np,
                self.data_store.inside_batch,
            )

        if not lf.ready:
            return i

        lf.walk()
        enhanced = lf.get_processed_planes()
        raw_data, masks = lf.get_processed_side_data_planes()

        self.data_store.peak_enhanced_batch = enhanced, raw_data, masks
        self.save_tiffs(
            "peak_enhanced",
            i,
            [p.cpu().numpy() for p in enhanced],
        )

        return i + len(enhanced)

    def batch_bin_2d_peaks(
        self,
        i,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ) -> None:
        if start_gen_from <= DetectionStage.bin_2d_peaks <= end_gen_on:
            enhanced, np_data, masks = self.data_store.peak_enhanced_batch
            bin_2d_peaks = self.plane_filter.threshold_peak_enhanced_planes(
                torch.stack(enhanced)
            )

            self.data_store.bin_2d_peaks_batch = (
                enhanced,
                list(bin_2d_peaks),
                np_data,
                masks,
            )
            self.save_tiffs("bin_2d_peaks", i, bin_2d_peaks.cpu().numpy())

    def batch_bin_3d_peaks(
        self,
        i,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
        do_flush: bool = False,
    ) -> int | None:
        tf = self.threshold_filter
        if tf is None:
            return None

        if not (start_gen_from <= DetectionStage.bin_3d_peaks <= end_gen_on):
            return None

        if do_flush:
            if not tf.flush():
                return i
        else:
            enhanced, bin_2d_peaks, np_data, masks = (
                self.data_store.bin_2d_peaks_batch
            )
            tf.append(
                enhanced,
                bin_2d_peaks,  # clone
                np_data,
                masks,
            )

        if not tf.ready:
            return i

        tf.walk()
        binarized = tf.get_processed_planes()
        raw_data, masks = tf.get_processed_side_data_planes()

        self.data_store.bin_3d_peaks_batch = binarized, raw_data, masks
        self.save_tiffs(
            "bin_3d_peaks",
            i,
            [p.cpu().numpy() for p in binarized],
        )

        return i + len(binarized)

    def batch_filter_3d_ball(
        self,
        i,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
        do_flush: bool = False,
    ) -> int | None:
        if not (
            start_gen_from <= DetectionStage.filtered_3d_ball <= end_gen_on
        ):
            return None

        bf = self.ball_filter
        tf = self.threshold_filter

        if do_flush:
            if not bf.flush():
                return i
        else:
            if tf is None:
                _, bin_peaks, np_input, inside = (
                    self.data_store.bin_2d_peaks_batch
                )
            else:
                bin_peaks, np_input, inside = (
                    self.data_store.bin_3d_peaks_batch
                )

            # bf.append(bin_peaks, inside, np_input)
            bf.use_mask = False
            bf.append(bin_peaks, None, np_input)

        if not bf.ready:
            return i

        bf.walk()
        data_planes = bf.get_processed_planes()
        raw_planes = bf.get_raw_planes()
        self.data_store.filtered_3d_ball_batch = data_planes, raw_planes

        for buff in data_planes:
            buff[buff != self.settings.soma_centre_value] = 0
        self.save_tiffs(
            "filtered_3d_ball",
            i,
            data_planes,
        )

        return i + len(raw_planes)

    def batch_structs_detection(
        self,
        i,
        previous_plane: np.ndarray | None,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ) -> tuple[np.ndarray | None, int]:
        if not (
            start_gen_from <= DetectionStage.structs_detection <= end_gen_on
        ):
            return previous_plane, i

        detection_converter = self.settings.detection_data_converter_func

        data_planes, raw_planes = self.data_store.filtered_3d_ball_batch

        for k, (plane, raw_plane) in enumerate(zip(data_planes, raw_planes)):
            previous_plane = self.cell_detector.process(
                detection_converter(plane), previous_plane, raw_plane
            )
            self.save_tiffs(
                "structs_id",
                i + k,
                previous_plane[None, :, :].astype(np.uint32),
            )

        return previous_plane, i + len(data_planes)

    def process_structures_type(
        self,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ):
        if not (start_gen_from <= DetectionStage.struct_typed <= end_gen_on):
            return

        min_split_size = self.settings.max_cell_volume
        max_split_size = self.settings.max_cluster_size
        detect_coi = self.settings.detect_centre_of_intensity
        cell_detector = self.cell_detector
        structs_pkl = []

        shape = self.signal_shape
        struct_type_vol = np.zeros(shape, dtype=np.uint8)

        cells = {
            "struct_cell_candidate": [],
            "struct_needs_split": [],
            "struct_too_big": [],
        }

        s_intensities = cell_detector.get_structures_intensities()
        n = cell_detector.n_structures
        for cell_id, arr in tqdm.tqdm(
            cell_detector.get_structures().items(),
            total=n,
            unit="structures",
            desc="Processing structures",
        ):
            structs_pkl.append((cell_id, arr, s_intensities[cell_id]))

            intensity = None
            if detect_coi:
                intensity = s_intensities[cell_id]

            struct_size = len(arr)
            cx, cy, cz = get_structure_centre(arr, intensity)

            if struct_size < min_split_size:
                tp = "struct_cell_candidate"
                color = 1
            elif struct_size < max_split_size:
                tp = "struct_needs_split"
                color = 2
            else:
                tp = "struct_too_big"
                color = 3

            metadata = {
                "struct_size": struct_size,
                "struct_id": cell_id,
                "struct_type": tp,
            }
            cells[tp].append(
                Cell((cx, cy, cz), Cell.UNKNOWN, metadata=metadata)
            )
            struct_type_vol[arr[:, 2], arr[:, 1], arr[:, 0]] = color

        with open(self.structures_data_path, "wb") as fh:
            pickle.dump(structs_pkl, fh, pickle.HIGHEST_PROTOCOL)

        self.save_tiffs("struct_type", 0, struct_type_vol)

        for name, item_cells in cells.items():
            save_cells(item_cells, str(self.local_store / f"{name}.yml"))

    def split_structs(
        self,
        structs_to_split,
        volume: np.ndarray,
        color: int,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> list[Cell]:
        next_sid = self.cell_detector.next_structure_id
        metadata = {
            "original_struct_id": 0,
            "struct_size": 0,
            "struct_id": 0,
            "struct_type": "struct_split_cell_candidate",
        }
        cells = []
        total_structs = len(structs_to_split)
        progress_bar = tqdm.tqdm(
            total=total_structs, desc="Splitting cell clusters"
        )

        f = partial(_split_cells, settings=self.splitting_settings)
        ctx = mp.get_context("spawn")
        # we can't use the context manager because of coverage issues:
        # https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
        pool = ctx.Pool(processes=self.settings.n_processes)
        i = 0
        try:
            for structures, did_split, orig_cid in pool.imap_unordered(
                f, structs_to_split
            ):
                metadata["original_struct_id"] = orig_cid
                for cid, (centre, arr) in structures.items():
                    metadata["struct_size"] = len(arr)
                    if did_split:
                        metadata["struct_id"] = next_sid
                        next_sid += 1
                    else:
                        assert len(structures) == 1
                        metadata["struct_id"] = cid

                    volume[arr[:, 2], arr[:, 1], arr[:, 0]] = color
                    cells.append(Cell(centre, Cell.UNKNOWN, metadata=metadata))

                if not i % 100:
                    progress_callback(total_structs, i, "Splitting cells")
                i += 1
                progress_bar.update()
        finally:
            pool.close()
            pool.join()

        progress_bar.close()

        return cells

    def process_structure_splitting(
        self,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
        progress_callback: Callable[[int, int, str], None] = None,
    ):
        if not (
            start_gen_from <= DetectionStage.struct_splitting <= end_gen_on
        ):
            return

        min_split_size = self.settings.max_cell_volume
        max_split_size = self.settings.max_cluster_size
        detect_coi = self.settings.detect_centre_of_intensity
        cell_detector = self.cell_detector

        shape = self.signal_shape
        struct_type_split_vol = np.zeros(shape, dtype=np.uint8)

        structs_to_split = []

        s_intensities = cell_detector.get_structures_intensities()
        n = cell_detector.n_structures
        for cell_id, arr in tqdm.tqdm(
            cell_detector.get_structures().items(),
            total=n,
            unit="structures",
            desc="Finding structures to split",
        ):
            struct_size = len(arr)

            if struct_size < min_split_size:
                color = 1
            elif struct_size < max_split_size:
                color = None
            else:
                color = 3

            if color is None:
                intensity = s_intensities[cell_id] if detect_coi else None
                structs_to_split.append((cell_id, arr, intensity))
            else:
                struct_type_split_vol[arr[:, 2], arr[:, 1], arr[:, 0]] = color

        struct_split_into_cell_candidate = self.split_structs(
            structs_to_split, struct_type_split_vol, 2, progress_callback
        )

        self.save_tiffs(
            "struct_type_split",
            0,
            struct_type_split_vol,
        )

        save_cells(
            struct_split_into_cell_candidate,
            str(self.local_store / "struct_split_into_cell_candidate.yml"),
        )

    def _run_filter(
        self,
        start_gen_from: DetectionStage = DetectionStage.input,
        end_gen_on: DetectionStage = DetectionStage.struct_splitting,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> None:
        n_planes = self.signal_shape[0]
        lf = self.lap_filter
        tf = self.threshold_filter
        bf = self.ball_filter

        self.local_store.mkdir(parents=True, exist_ok=True)
        save_yaml(
            {
                "settings": self.settings,
                "splitting_settings": self.splitting_settings,
                "input_settings": self.input_settings,
            },
            self.config_yaml_path,
        )

        n_enhanced = 0
        n_3d_bin = 0
        n_ball = 0
        n_detected = 0
        previous_plane = None
        for i in tqdm.tqdm(range(0, n_planes, self.batch_size), unit="planes"):
            self.data_store.set_current_batch(i)
            if progress_callback is not None:
                progress_callback(n_planes, i, "Detecting cells")

            self.batch_input(i, start_gen_from, end_gen_on)
            self.batch_clip(i, start_gen_from, end_gen_on)
            processed = self.batch_peak_enhanced(
                n_enhanced, start_gen_from, end_gen_on
            )
            if processed is None:
                self.batch_bin_2d_peaks(i, start_gen_from, end_gen_on)
            else:
                if processed == n_enhanced:
                    assert not lf.ready
                    continue

                assert lf.ready
                self.batch_bin_2d_peaks(n_enhanced, start_gen_from, end_gen_on)
                n_enhanced = processed

            processed = self.batch_bin_3d_peaks(
                n_3d_bin, start_gen_from, end_gen_on
            )
            if processed is not None:
                assert tf is not None
                if processed == n_3d_bin:
                    assert not tf.ready
                    continue

                assert tf.ready
                n_3d_bin = processed

            processed = self.batch_filter_3d_ball(
                n_ball,
                start_gen_from,
                end_gen_on,
            )
            if processed is not None:
                if processed == n_ball:
                    assert not bf.ready
                    continue

                assert bf.ready
                n_ball = processed

            previous_plane, n_detected = self.batch_structs_detection(
                n_detected,
                previous_plane,
                start_gen_from,
                end_gen_on,
            )

            try:
                # try to get any errors from the thread, if present
                msg = self.thread_runner.get_msg_from_thread(timeout=0)
                if msg == EOFSignal:
                    raise TypeError("Tiff saving thread exited early")
            except Empty:
                pass

        processed = self.batch_peak_enhanced(
            n_enhanced, start_gen_from, end_gen_on, do_flush=True
        )
        if processed is not None and processed != n_enhanced:
            self.batch_bin_2d_peaks(n_enhanced, start_gen_from, end_gen_on)

            processed = self.batch_bin_3d_peaks(
                n_3d_bin, start_gen_from, end_gen_on
            )
            if processed is not None and n_3d_bin != processed:
                n_3d_bin = processed
                # there was new data in the tf flush, add it to ball
                processed = self.batch_filter_3d_ball(
                    n_ball,
                    start_gen_from,
                    end_gen_on,
                )
                if processed is not None and n_ball != processed:
                    # there was new data in ball, process it
                    n_ball = processed
                    previous_plane, n_detected = self.batch_structs_detection(
                        n_detected,
                        previous_plane,
                        start_gen_from,
                        end_gen_on,
                    )

        processed = self.batch_bin_3d_peaks(
            n_3d_bin, start_gen_from, end_gen_on, do_flush=True
        )
        if processed is not None and n_3d_bin != processed:
            # there was new data in the tf flush, add it to ball
            processed = self.batch_filter_3d_ball(
                n_ball,
                start_gen_from,
                end_gen_on,
            )
            if processed is not None and n_ball != processed:
                # there was new data in ball, process it
                n_ball = processed
                previous_plane, n_detected = self.batch_structs_detection(
                    n_detected,
                    previous_plane,
                    start_gen_from,
                    end_gen_on,
                )

        processed = self.batch_filter_3d_ball(
            n_ball,
            start_gen_from,
            end_gen_on,
            do_flush=True,
        )
        if processed is not None and n_ball != processed:
            # there was new data in ball flush, process it
            self.batch_structs_detection(
                n_detected,
                previous_plane,
                start_gen_from,
                end_gen_on,
            )

        self.process_structures_type(start_gen_from, end_gen_on)
        self.process_structure_splitting(
            start_gen_from, end_gen_on, progress_callback
        )

        # reset caches
        self.data_store = None
        if self.torch_device != "cpu":
            torch.cuda.empty_cache()

    @inference_wrapper
    def run_filter(
        self,
        start_gen_from: DetectionStage = DetectionStage.input,
        end_gen_on: DetectionStage = DetectionStage.struct_splitting,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> None:
        self.thread_runner = ThreadWithException(
            target=thread_run, pass_self=True
        )
        thread_runner = self.thread_runner
        thread_runner.start()

        try:
            self._run_filter(start_gen_from, end_gen_on, progress_callback)
        finally:
            thread_runner.notify_to_end_thread()
            self.thread_runner = None

            thread_runner.join()
            thread_runner.clear_remaining()


@inference_wrapper
def _split_cells(
    arg, settings: DetectionSettings
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], bool, int]:
    # runs in its own process for a bright region to be split.
    # For splitting cells, we only run with one thread. Because the volume is
    # likely small and using multiple threads would cost more in overhead than
    # is worth. num threads can be set only at processes level.
    torch.set_num_threads(1)
    cell_id, cell_points, intensity = arg
    try:
        centers, detector = split_cells(
            cell_points, settings=settings, intensity=intensity
        )
        if detector is None:
            return {cell_id: (centers[0, :], cell_points)}, False, cell_id

        structures = {}
        cell_detector_split, offset = detector
        s_intensities_split = cell_detector_split.get_structures_intensities()
        for (
            cell_id_split,
            arr_split,
        ) in cell_detector_split.get_structures().items():
            arr_split += offset[None, :]

            intensity_split = None
            if settings.detect_centre_of_intensity:
                intensity_split = s_intensities_split[cell_id_split]

            center_split = get_structure_centre(arr_split, intensity_split)
            structures[cell_id_split] = center_split, arr_split

        return structures, True, cell_id
    except (ValueError, AssertionError) as err:
        raise StructureSplitException(f"Cell {cell_id}, error; {err}")


if __name__ == "__main__":
    with torch.inference_mode(True):
        detection_debug = DetectionDebug(
            signal_shape=tifffile.memmap(
                r"D:\code_data\lightsheet\cellfinder\MF1_378M_W_BS_561_cropped.tif"
            ).shape,
            local_store=r"D:\code_data\lightsheet\cellfinder\store",
            batch_size=1,
            torch_device="cuda",
            voxel_sizes=(4, 2.03, 2.03),
            soma_diameter=8,
            max_cluster_size=1_000_000,
            ball_xy_size=10,
            ball_z_size=10,
            ball_overlap_fraction=0.6,  # 0.6 - 0.65
            soma_spread_factor=1.5,
            n_free_cpus=2,
            log_sigma_size=0.2,
            n_sds_above_mean_thresh=0.6,
            detect_centre_of_intensity=True,
            split_ball_xy_size=10,
            split_ball_z_size=10,
            split_ball_overlap_fraction=0.6,
            n_splitting_iter=10,
            n_sds_above_mean_tiled_thresh=1.25,
            tiled_thresh_tile_size=5,
        )
        detection_debug.load_data(
            signal=r"D:\code_data\lightsheet\cellfinder\MF1_378M_W_BS_561_cropped.tif",
            start_gen_from=DetectionStage.input,
            end_gen_on=DetectionStage.struct_splitting,
        )
        detection_debug.run_filter(
            start_gen_from=DetectionStage.input,
            end_gen_on=DetectionStage.struct_splitting,
        )

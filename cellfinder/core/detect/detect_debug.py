import dataclasses
import math
import pickle
from enum import IntEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Callable, Type

import numpy as np
import tifffile
import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.IO.cells import save_cells
from brainglobe_utils.IO.image.load import read_z_stack

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter import BallFilter
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    split_cells,
)


class DetectionStage(IntEnum):

    input = auto()
    clipped = auto()
    enhanced = auto()
    filtered_2d = auto()
    filtered_3d = auto()
    splitting = auto()


class DetectionDebug:
    """Saves everything to the data store."""

    settings: DetectionSettings

    splitting_settings: DetectionSettings

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
        return self.local_store / "struct_cell_candidate.xml"

    @property
    def structs_needs_split_path(self) -> Path:
        return self.local_store / "struct_needs_split.xml"

    @property
    def structs_too_big_path(self) -> Path:
        return self.local_store / "struct_too_big.xml"

    @property
    def struct_split_into_cell_candidate_path(self) -> Path:
        return self.local_store / "struct_split_into_cell_candidate.xml"

    @property
    def input_image_path(self) -> Path:
        return self.local_store / "input"

    @property
    def clipped_input_image_path(self) -> Path:
        return self.local_store / "clipped"

    @property
    def enhanced_image_path(self) -> Path:
        return self.local_store / "enhanced"

    @property
    def inside_brain_image_path(self) -> Path:
        return self.local_store / "inside_brain"

    @property
    def filtered_2d_image_path(self) -> Path:
        return self.local_store / "filtered_2d"

    @property
    def filtered_3d_image_path(self) -> Path:
        return self.local_store / "filtered_3d"

    @property
    def structs_id_image_path(self) -> Path:
        return self.local_store / "struct_id"

    @property
    def struct_type_image_path(self) -> Path:
        return self.local_store / "struct_type"

    @property
    def struct_type_split_image_path(self) -> Path:
        return self.local_store / "struct_type_split"

    @cached_property
    def tile_processor(self) -> TileProcessor:
        settings = self.settings
        return TileProcessor(
            plane_shape=self.signal_shape[1:],
            clipping_value=settings.clipping_value,
            threshold_value=settings.threshold_value,
            n_sds_above_mean_thresh=settings.n_sds_above_mean_thresh,
            n_sds_above_mean_tiled_thresh=settings.n_sds_above_mean_tiled_thresh,
            tiled_thresh_tile_size=settings.tiled_thresh_tile_size,
            soma_diameter=settings.soma_diameter,
            log_sigma_size=settings.log_sigma_size,
            torch_device=self.torch_device,
            dtype=settings.filtering_dtype.__name__,
            use_scipy=self.use_scipy,
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
            start_z=self.ball_filter.first_valid_plane,
            soma_centre_value=settings.detection_soma_centre_value,
        )

    def save_tiffs(
        self,
        prefix: str,
        start_index: int,
        buffer: torch.Tensor | np.ndarray,
    ):
        root = self.local_store / prefix
        root.mkdir(parents=True, exist_ok=True)

        if isinstance(buffer, torch.Tensor):
            arr = buffer.cpu().numpy()
        else:
            arr = buffer

        digits = int(math.ceil(math.log10(self.signal_shape[0])))
        for i, plane in enumerate(arr, start_index):
            tifffile.imwrite(
                root / f"{prefix}_{i:0{digits}}.tif",
                plane,
                compression="LZW",
            )

    def pad_3d_filtered_images(
        self,
        sample_plane: torch.Tensor | np.ndarray,
        n_saved_planes: int,
    ):
        """
        3d filters skip the first and last planes. This pads them by creating
        those planes as blank planes so that all outputs have same number of
        planes.
        """
        n = self.signal_shape[0]

        if self.ball_filter.first_valid_plane:
            # 3d filters skip first few planes
            buff = np.zeros(
                (self.ball_filter.first_valid_plane, *sample_plane.shape),
                dtype=sample_plane.dtype,
            )
            self.save_tiffs("filtered_3d", 0, buff)
            self.save_tiffs(
                "struct_id",
                0,
                buff.astype(np.uint32),
            )

            n_saved_planes += self.ball_filter.first_valid_plane

        if n_saved_planes < n:
            buff = np.zeros(
                (n - n_saved_planes, *sample_plane.shape),
                dtype=sample_plane.dtype,
            )
            self.save_tiffs("filtered_3d", n_saved_planes, buff)
            self.save_tiffs(
                "struct_id", n_saved_planes, buff.astype(np.uint32)
            )

    def _load_signal_data(
        self,
        signal: Path | str | np.ndarray | None,
        start_plane: int,
        end_plane: int,
    ):
        signal_array = signal
        if not isinstance(signal, np.ndarray):
            signal_array = read_z_stack(str(signal))

        if end_plane <= 0:
            end_plane = len(signal_array)
        signal_array = signal_array[start_plane:end_plane, :, :]

        if signal_array.shape != self.signal_shape:
            raise ValueError(
                f"Expected signal with shape {self.signal_shape}, "
                f"got {signal_array.shape}"
            )

        signal_array = np.asarray(signal_array).astype(
            self.settings.plane_original_np_dtype
        )
        signal_array = self.settings.filter_data_converter_func(signal_array)
        signal_array_torch = torch.from_numpy(signal_array).to(
            self.torch_device
        )

        return signal_array, signal_array_torch

    def _read_image_from_local_store(
        self, prefix: str, dtype: Type | None = None
    ):
        data = read_z_stack(str(self.local_store / prefix))
        if data.shape[0] != self.signal_shape[0]:
            raise ValueError(
                f"Expected {self.signal_shape[0]} planes from {prefix}, "
                f"got {data.shape[0]}"
            )

        data = np.asarray(data, dtype=dtype)
        return data

    def _load_data(
        self,
        start_from_stage: DetectionStage,
        end_on_stage: DetectionStage,
        signal: Path | str | np.ndarray | None = None,
        start_plane: int = 0,
        end_plane: int = 0,
    ):
        filter_dtype = self.settings.filtering_dtype

        # we always need the input signal data for all stages. But we load from
        # input only if we start at input
        self.signal_array_np = None
        self.signal_array_torch = None
        if start_from_stage == DetectionStage.input <= end_on_stage:
            if signal is None:
                raise ValueError("Input data not provided")
            sig = signal
        else:
            sig = self._read_image_from_local_store("input")

        signal_array_np, signal_array = self._load_signal_data(
            sig, start_plane, end_plane
        )
        self.signal_array_np = signal_array_np
        self.signal_array_torch = signal_array

        # we only use clipped data when doing enhancement filter. If starting
        # before enhancement we use the generated clipped data directly
        self.clipped_data = None
        if start_from_stage == DetectionStage.enhanced <= end_on_stage:
            clipped_data = self._read_image_from_local_store(
                "clipped", filter_dtype
            )
            clipped_data = torch.from_numpy(clipped_data).to(self.torch_device)
            self.clipped_data = clipped_data

        # we only use 2d filtered data when doing 3d filter. If starting
        # before 3d filtering we use the generated 2d data directly
        self.inside_data = None
        self.filtered_2d_data = None
        if start_from_stage == DetectionStage.filtered_3d <= end_on_stage:
            inside_data = self._read_image_from_local_store(
                "inside_brain", bool
            )
            inside_data = torch.from_numpy(inside_data).to(self.torch_device)
            self.inside_data = inside_data

            filtered_2d = self._read_image_from_local_store(
                "filtered_2d", filter_dtype
            )
            filtered_2d = torch.from_numpy(filtered_2d).to(self.torch_device)
            self.filtered_2d_data = filtered_2d

        # if starting from before splitting, we generate the cell data anew
        if start_from_stage == DetectionStage.splitting <= end_on_stage:
            cell_detector = self.cell_detector
            with open(self.structures_data_path, "rb") as fh:
                structs = pickle.load(fh)

            for sid, arr, intensity in structs:
                cell_detector.add_points(sid, arr, intensity)

    def batch_input(
        self, i, start_from_stage: DetectionStage, end_on_stage: DetectionStage
    ):
        batch_size = self.batch_size

        batch_torch = self.signal_array_torch[i : i + batch_size]
        batch_np = self.signal_array_np[i : i + batch_size]

        if start_from_stage <= DetectionStage.input <= end_on_stage:
            self.save_tiffs("input", i, batch_torch)

        return batch_torch, batch_np

    def batch_clip(
        self,
        i,
        batch_torch,
        start_from_stage: DetectionStage,
        end_on_stage: DetectionStage,
    ):
        if start_from_stage <= DetectionStage.clipped <= end_on_stage:
            batch_clipped = torch.clone(batch_torch)
            torch.clip_(batch_clipped, 0, self.tile_processor.clipping_value)
            self.save_tiffs("clipped", i, batch_clipped)
        elif start_from_stage == DetectionStage.enhanced <= end_on_stage:
            batch_clipped = self.clipped_data[i : i + self.batch_size]
        else:
            batch_clipped = None

        return batch_clipped

    def batch_enhanced(
        self,
        i,
        batch_clipped,
        start_from_stage: DetectionStage,
        end_on_stage: DetectionStage,
    ):
        if start_from_stage <= DetectionStage.enhanced <= end_on_stage:
            enhanced_planes = self.tile_processor.peak_enhancer.enhance_peaks(
                batch_clipped
            )
            self.save_tiffs("enhanced", i, enhanced_planes)

    def batch_filter_2d(
        self,
        i,
        batch_torch,
        start_from_stage: DetectionStage,
        end_on_stage: DetectionStage,
    ):
        if start_from_stage <= DetectionStage.filtered_2d <= end_on_stage:
            filtered_2d, inside_brain_tiles = (
                self.tile_processor.get_tile_mask(batch_torch)
            )
            self.save_tiffs("inside_brain", i, inside_brain_tiles)
            self.save_tiffs("filtered_2d", i, filtered_2d)
        elif start_from_stage == DetectionStage.filtered_3d <= end_on_stage:
            inside_brain_tiles = self.inside_data[i : i + self.batch_size]
            filtered_2d = self.filtered_2d_data[i : i + self.batch_size]
        else:
            filtered_2d = None
            inside_brain_tiles = None

        return filtered_2d, inside_brain_tiles

    def batch_filter_3d(
        self,
        i,
        filtered_2d,
        inside_brain_tiles,
        batch_np,
        n_3d_planes: int,
        previous_plane: np.ndarray,
        start_from_stage: DetectionStage,
        end_on_stage: DetectionStage,
    ):
        middle_planes = None
        if (
            start_from_stage > DetectionStage.filtered_3d
            or end_on_stage < DetectionStage.filtered_3d
        ):
            return n_3d_planes, previous_plane, middle_planes

        ball_filter = self.ball_filter
        detection_converter = self.settings.detection_data_converter_func

        ball_filter.append(filtered_2d, inside_brain_tiles, batch_np)
        if ball_filter.ready:
            ball_filter.walk()
            middle_planes = ball_filter.get_processed_planes()
            raw_planes = ball_filter.get_raw_planes()
            buff = middle_planes.copy()
            buff[buff != self.settings.soma_centre_value] = 0
            self.save_tiffs(
                "filtered_3d",
                n_3d_planes + ball_filter.first_valid_plane,
                buff,
            )

            detection_middle_planes = detection_converter(middle_planes)

            for k, (plane, raw_plane, detection_plane) in enumerate(
                zip(middle_planes, raw_planes, detection_middle_planes)
            ):
                previous_plane = self.cell_detector.process(
                    detection_plane, previous_plane, raw_plane
                )
                self.save_tiffs(
                    "struct_id",
                    i + k + ball_filter.first_valid_plane,
                    previous_plane[None, :, :].astype(np.uint32),
                )

            n_3d_planes += len(middle_planes)

        return n_3d_planes, previous_plane, middle_planes

    def process_structures(
        self,
        start_from_stage: DetectionStage,
        end_on_stage: DetectionStage,
    ):
        if (
            start_from_stage > DetectionStage.splitting
            or end_on_stage < DetectionStage.splitting
        ):
            return

        min_split_size = self.settings.max_cell_volume
        max_split_size = self.settings.max_cluster_size
        detect_coi = self.settings.detect_centre_of_intensity
        cell_detector = self.cell_detector
        next_sid = cell_detector.next_structure_id
        structs_pkl = []

        shape = self.signal_shape
        struct_type_vol = np.zeros(shape, dtype=np.uint8)
        struct_type_split_vol = np.zeros(shape, dtype=np.uint8)

        cells = {
            "struct_cell_candidate": [],
            "struct_needs_split": [],
            "struct_too_big": [],
            "struct_split_into_cell_candidate": [],
        }
        s_intensities = cell_detector.get_structures_intensities()
        for cell_id, arr in cell_detector.get_structures().items():
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

            if tp == "needs_split":
                metadata["struct_type"] = "struct_split_cell_candidate"

                centers, detector = split_cells(
                    arr, self.splitting_settings, intensity
                )
                if detector is None:
                    struct_type_split_vol[arr[:, 2], arr[:, 1], arr[:, 0]] = (
                        color
                    )
                    cells["struct_split_into_cell_candidate"].append(
                        Cell((cx, cy, cz), Cell.UNKNOWN, metadata=metadata)
                    )
                else:
                    cell_detector_split, offset = detector

                    s_intensities_split = (
                        cell_detector_split.get_structures_intensities()
                    )
                    for (
                        cell_id_split,
                        arr_split,
                    ) in cell_detector_split.get_structures().items():
                        struct_type_split_vol[
                            arr_split[:, 2], arr_split[:, 1], arr_split[:, 0]
                        ] = color

                        intensity_split = None
                        if detect_coi:
                            intensity_split = s_intensities_split[
                                cell_id_split
                            ]

                        cx, cy, cz = get_structure_centre(
                            arr_split, intensity_split
                        )

                        metadata["struct_size"] = len(arr_split)
                        metadata["struct_id"] = next_sid
                        next_sid += 1
                        cells["struct_split_into_cell_candidate"].append(
                            Cell((cx, cy, cz), Cell.UNKNOWN, metadata=metadata)
                        )
            else:
                struct_type_split_vol[arr[:, 2], arr[:, 1], arr[:, 0]] = color

        with open(self.structures_data_path, "wb") as fh:
            pickle.dump(structs_pkl, fh, pickle.HIGHEST_PROTOCOL)

        self.save_tiffs("struct_type", 0, struct_type_vol)
        self.save_tiffs(
            "struct_type_split",
            0,
            struct_type_split_vol,
        )

        for name, item_cells in cells.items():
            save_cells(item_cells, str(self.local_store / f"{name}.xml"))

    def run_filter(
        self,
        start_from_stage: DetectionStage = DetectionStage.input,
        end_on_stage: DetectionStage = DetectionStage.splitting,
        signal: Path | str | np.ndarray | None = None,
        start_plane: int = 0,
        end_plane: int = 0,
        progress_callback: Callable[[int], None] = None,
    ):
        self._load_data(
            start_from_stage, end_on_stage, signal, start_plane, end_plane
        )

        previous_plane = None
        n_3d_planes = 0
        middle_planes = None

        for i in tqdm.tqdm(range(0, self.signal_shape[0], self.batch_size)):
            batch_torch, batch_np = self.batch_input(
                i, start_from_stage, end_on_stage
            )
            batch_clipped = self.batch_clip(
                i, batch_torch, start_from_stage, end_on_stage
            )
            self.batch_enhanced(
                i, batch_clipped, start_from_stage, end_on_stage
            )
            filtered_2d, inside_brain_tiles = self.batch_filter_2d(
                i, batch_torch, start_from_stage, end_on_stage
            )
            n_3d_planes, previous_plane, middle_planes = self.batch_filter_3d(
                i,
                filtered_2d,
                inside_brain_tiles,
                batch_np,
                n_3d_planes,
                previous_plane,
                start_from_stage,
                end_on_stage,
            )

            if progress_callback is not None:
                progress_callback(i)

        if start_from_stage <= DetectionStage.filtered_3d <= end_on_stage:
            self.pad_3d_filtered_images(
                middle_planes[0, :, :],
                n_3d_planes,
            )

        self.process_structures(start_from_stage, end_on_stage)


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
        detection_debug.run_filter(
            signal=r"D:\code_data\lightsheet\cellfinder\MF1_378M_W_BS_561_cropped.tif",
            start_from_stage=DetectionStage.splitting,
            end_on_stage=DetectionStage.splitting,
        )

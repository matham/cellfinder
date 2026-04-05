import dataclasses
import math
import multiprocessing as mp
import pickle
from enum import IntEnum, auto
from functools import cached_property, partial
from pathlib import Path
from typing import Callable, Type

import numpy as np
import tifffile
import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.IO.cells import save_cells
from brainglobe_utils.IO.image.load import read_z_stack
from brainglobe_utils.IO.yaml import save_yaml

from cellfinder.core import types
from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter import BallFilter
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)
from cellfinder.core.tools.tools import inference_wrapper


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

    input_settings: dict

    signal_array: types.array | None = None

    clipped_data: types.array | None = None

    inside_data: types.array | None = None

    filtered_2d_data: types.array | None = None

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

    @property
    def config_yaml_path(self) -> Path:
        return self.local_store / "detection_config.yml"

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

        if data.shape[0] != self.signal_shape[0]:
            raise ValueError(
                f"Expected {self.signal_shape[0]} planes, got {data.shape[0]}"
            )

        if dtype is not None:
            data = data.astype(dtype)

        return data

    def _crop_image_if_needed(
        self,
        data: types.array,
        start_plane: int = 0,
        end_plane: int = 0,
        bottom_corner: tuple[int, int] = (0, 0),
        top_corner: tuple[int, int] = (0, 0),
    ) -> types.array:
        bot_y, bot_x = bottom_corner
        top_y, top_x = top_corner

        start_plane = max(start_plane, 0)
        bot_y = max(bot_y, 0)
        bot_x = max(bot_x, 0)

        if end_plane <= 0:
            end_plane = data.shape[0]
        if top_y <= 0:
            top_y = data.shape[1]
        if top_x <= 0:
            top_x = data.shape[2]

        return data[start_plane:end_plane, bot_y:top_y, bot_x:top_x]

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
        end_gen_on: DetectionStage = DetectionStage.splitting,
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
                if signal is None or isinstance(signal, np.ndarray)
                else str(signal)
            ),
        }

        crop_f = partial(
            self._crop_image_if_needed,
            start_plane=start_plane,
            end_plane=end_plane,
            bottom_corner=bottom_corner,
            top_corner=top_corner,
        )

        # we need the input signal data for all filtering stages except
        # splitting. But we load from input only if we start at input
        self.signal_array = None
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

            self.signal_array = crop_f(data)
        elif start_gen_from < DetectionStage.splitting:
            if local_store_loader is None:
                data = self.get_image(local_prefix="input")
            else:
                data = self.get_image(
                    arr_or_path=local_store_loader.input_image_path
                )

            self.signal_array = crop_f(data)

        # we only use clipped data when passing input to enhancement filter.
        # If starting before enhancement we use the generated data
        self.clipped_data = None
        if start_gen_from == DetectionStage.enhanced <= end_gen_on:
            if local_store_loader is None:
                data = self.get_image(local_prefix="clipped")
            else:
                data = self.get_image(
                    arr_or_path=local_store_loader.clipped_input_image_path
                )
            self.clipped_data = crop_f(data)

        # we only use 2d filtered data when passing input to the 3d filter. If
        # starting before 3d filtering we use the generated 2d data directly
        self.inside_data = None
        self.filtered_2d_data = None
        if start_gen_from == DetectionStage.filtered_3d <= end_gen_on:
            if local_store_loader is None:
                inside_data = self.get_image(local_prefix="inside_brain")
                filtered_data = self.get_image(local_prefix="filtered_2d")
            else:
                inside_data = self.get_image(
                    arr_or_path=local_store_loader.inside_brain_image_path
                )
                filtered_data = self.get_image(
                    arr_or_path=local_store_loader.filtered_2d_image_path
                )
            self.inside_data = crop_f(inside_data)
            self.filtered_2d_data = crop_f(filtered_data)

        # if starting from before splitting, we generate the cell data anew so
        # only load existing data if starting at splitting
        if start_gen_from == DetectionStage.splitting <= end_gen_on:
            self._load_detected_structs(
                local_store_loader,
                start_plane,
                end_plane,
                bottom_corner,
                top_corner,
            )

    def batch_input(
        self, i, start_gen_from: DetectionStage, end_gen_on: DetectionStage
    ):
        batch_size = self.batch_size

        batch_np = np.asarray(self.signal_array[i : i + batch_size]).astype(
            self.settings.plane_original_np_dtype
        )
        batch_np = self.settings.filter_data_converter_func(batch_np)
        batch_torch = torch.from_numpy(batch_np).to(self.torch_device)

        if start_gen_from <= DetectionStage.input <= end_gen_on:
            self.save_tiffs("input", i, batch_np)

        return batch_torch, batch_np

    def batch_clip(
        self,
        i,
        batch_torch,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ):
        if start_gen_from <= DetectionStage.clipped <= end_gen_on:
            batch_clipped = torch.clone(batch_torch)
            torch.clip_(batch_clipped, 0, self.tile_processor.clipping_value)
            self.save_tiffs("clipped", i, batch_clipped)
        elif start_gen_from == DetectionStage.enhanced <= end_gen_on:
            batch_clipped = self.clipped_data[i : i + self.batch_size]
            batch_clipped = np.asarray(
                batch_clipped, dtype=self.settings.filtering_dtype
            )
            batch_clipped = torch.from_numpy(batch_clipped).to(
                self.torch_device
            )
        else:
            batch_clipped = None

        return batch_clipped

    def batch_enhanced(
        self,
        i,
        batch_clipped,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ):
        if start_gen_from <= DetectionStage.enhanced <= end_gen_on:
            enhanced_planes = self.tile_processor.peak_enhancer.enhance_peaks(
                batch_clipped
            )
            self.save_tiffs("enhanced", i, enhanced_planes)

    def batch_filter_2d(
        self,
        i,
        batch_torch,
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ):
        if start_gen_from <= DetectionStage.filtered_2d <= end_gen_on:
            filtered_2d, inside_brain_tiles = (
                self.tile_processor.get_tile_mask(batch_torch)
            )
            self.save_tiffs("inside_brain", i, inside_brain_tiles)
            self.save_tiffs("filtered_2d", i, filtered_2d)
        elif start_gen_from == DetectionStage.filtered_3d <= end_gen_on:
            inside_brain_tiles = self.inside_data[i : i + self.batch_size]
            inside_brain_tiles = np.asarray(inside_brain_tiles, dtype=bool)
            inside_brain_tiles = torch.from_numpy(inside_brain_tiles).to(
                self.torch_device
            )

            filtered_2d = self.filtered_2d_data[i : i + self.batch_size]
            filtered_2d = np.asarray(
                filtered_2d, dtype=self.settings.filtering_dtype
            )
            filtered_2d = torch.from_numpy(filtered_2d).to(self.torch_device)
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
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ):
        middle_planes = None
        if not (start_gen_from <= DetectionStage.filtered_3d <= end_gen_on):
            return n_3d_planes, previous_plane, middle_planes

        ball_filter = self.ball_filter
        detection_converter = self.settings.detection_data_converter_func

        # ball_filter.append(filtered_2d, inside_brain_tiles, batch_np)
        ball_filter.inside_brain_tiles = None
        ball_filter.append(filtered_2d, None, batch_np)
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
        start_gen_from: DetectionStage,
        end_gen_on: DetectionStage,
    ):
        if not (start_gen_from <= DetectionStage.filtered_3d <= end_gen_on):
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
        if not (start_gen_from <= DetectionStage.splitting <= end_gen_on):
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

    def run_filter(
        self,
        start_gen_from: DetectionStage = DetectionStage.input,
        end_gen_on: DetectionStage = DetectionStage.splitting,
        progress_callback: Callable[[int, int, str], None] = None,
    ):
        previous_plane = None
        n_3d_planes = 0
        middle_planes = None
        n_planes = self.signal_shape[0]

        self.local_store.mkdir(parents=True, exist_ok=True)
        save_yaml(
            {
                "settings": self.settings,
                "splitting_settings": self.splitting_settings,
                "input_settings": self.input_settings,
            },
            self.config_yaml_path,
        )

        for i in tqdm.tqdm(range(0, n_planes, self.batch_size), unit="planes"):
            batch_torch, batch_np = self.batch_input(
                i, start_gen_from, end_gen_on
            )
            batch_clipped = self.batch_clip(
                i, batch_torch, start_gen_from, end_gen_on
            )
            self.batch_enhanced(i, batch_clipped, start_gen_from, end_gen_on)
            filtered_2d, inside_brain_tiles = self.batch_filter_2d(
                i, batch_torch, start_gen_from, end_gen_on
            )
            n_3d_planes, previous_plane, middle_planes = self.batch_filter_3d(
                i,
                filtered_2d,
                inside_brain_tiles,
                batch_np,
                n_3d_planes,
                previous_plane,
                start_gen_from,
                end_gen_on,
            )

            if progress_callback is not None:
                progress_callback(n_planes, i, "Detecting cells")

        if start_gen_from <= DetectionStage.filtered_3d <= end_gen_on:
            self.pad_3d_filtered_images(
                middle_planes[0, :, :],
                n_3d_planes,
            )

        self.process_structures(start_gen_from, end_gen_on)
        self.process_structure_splitting(
            start_gen_from, end_gen_on, progress_callback
        )


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
            end_gen_on=DetectionStage.splitting,
        )
        detection_debug.run_filter(
            start_gen_from=DetectionStage.input,
            end_gen_on=DetectionStage.splitting,
        )

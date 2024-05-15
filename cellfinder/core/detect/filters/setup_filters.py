from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Tuple

import numpy as np
from brainglobe_utils.general.system import get_num_processes

from cellfinder.core.tools.tools import (
    get_data_converter,
    get_max_possible_int_value,
)

# as seen in the benchmarks in the original PR, when running on CPU using
# more than ~12 cores it starts to result in slowdowns. So limit to 12 cores
# when doing computational work (e.g. torch.functional.Conv2D)
MAX_TORCH_COMP_THREADS = 12


@dataclass
class DetectionSettings:
    """

    plane_width, plane_height :
        Width/height of the planes.
    ball_xy_size :
        Diameter of the spherical kernel in the x/y dimensions.
    ball_z_size :
        Diameter of the spherical kernel in the z dimension.
        Equal to the number of planes that stacked to filter
        the central plane of the stack.
    overlap_fraction :
        The fraction of pixels within the spherical kernel that
        have to be over *threshold_value* for a pixel to be marked
        as having a high intensity.
    tile_step_width, tile_step_height :
        Width/height of individual tiles in the mask generated by
        2D filtering.
    threshold_value :
        Value above which an individual pixel is considered to have
        a high intensity.
    soma_centre_value :
        Value used to mark pixels with a high enough intensity.


    n_iter (int): The number of iterations to perform. Default is 10.
    """

    plane_shape: Tuple[int, int]
    plane_original_np_dtype: np.dtype
    filtering_dtype: str

    voxel_sizes: Tuple[float, float, float]
    soma_spread_factor: float
    soma_diameter_um: float
    max_cluster_size_um3: float
    ball_xy_size_um: float
    ball_z_size_um: float

    start_plane: int
    end_plane: int
    n_planes: int

    n_free_cpus: int

    ball_overlap_fraction: float
    log_sigma_size: float
    n_sds_above_mean_thresh: float

    outlier_keep: bool = False
    artifact_keep: bool = False

    save_planes: bool = False
    plane_directory: Optional[str] = None

    batch_size: int = 1
    torch_device: str = "cpu"

    num_prefetch_batches: int = 2

    n_splitting_iter: int = 10

    def __getstate__(self):
        d = self.__dict__.copy()
        # when sending across processes, we need to be able to pickle. This
        # property cannot be pickled (and doesn't need to be)
        if "filter_data_converter_func" in d:
            del d["filter_data_converter_func"]
        return d

    @cached_property
    def filter_data_converter_func(self) -> Callable[[np.ndarray], np.ndarray]:
        return get_data_converter(
            self.plane_original_np_dtype, getattr(np, self.filtering_dtype)
        )

    @cached_property
    def detection_dtype(self) -> np.dtype:
        working_dtype = getattr(np, self.filtering_dtype)
        if np.issubdtype(working_dtype, np.integer):
            # already integer, return it
            return working_dtype

        max_int = get_max_possible_int_value(working_dtype)
        if max_int <= get_max_possible_int_value(np.uint32):
            return np.uint32
        return np.uint64

    @cached_property
    def clipping_value(self) -> int:
        return (
            get_max_possible_int_value(getattr(np, self.filtering_dtype)) - 2
        )

    @cached_property
    def threshold_value(self) -> int:
        return (
            get_max_possible_int_value(getattr(np, self.filtering_dtype)) - 1
        )

    @cached_property
    def soma_centre_value(self) -> int:
        return get_max_possible_int_value(getattr(np, self.filtering_dtype))

    @property
    def tile_height(self) -> int:
        return self.soma_diameter * 2

    @property
    def tile_width(self) -> int:
        return self.soma_diameter * 2

    @property
    def plane_height(self) -> int:
        return self.plane_shape[0]

    @property
    def plane_width(self) -> int:
        return self.plane_shape[1]

    @property
    def n_processes(self) -> int:
        n = get_num_processes(min_free_cpu_cores=self.n_free_cpus)
        return max(n - 1, 1)

    @property
    def n_torch_comp_threads(self) -> int:
        # Reserve batch_size cores for batch parallelization on CPU, 1 per
        # plane. for GPU it doesn't matter either way because it doesn't use
        # threads. Also reserve for data feeding thread and cell detection
        n = max(1, self.n_processes - self.batch_size - 2)
        return min(n, MAX_TORCH_COMP_THREADS)

    @cached_property
    def soma_diameter(self) -> int:
        voxel_sizes = self.voxel_sizes
        mean_in_plane_pixel_size = 0.5 * (
            float(voxel_sizes[2]) + float(voxel_sizes[1])
        )
        return int(round(self.soma_diameter_um / mean_in_plane_pixel_size))

    @cached_property
    def max_cluster_size(self) -> int:
        voxel_sizes = self.voxel_sizes
        voxel_volume = (
            float(voxel_sizes[2])
            * float(voxel_sizes[1])
            * float(voxel_sizes[0])
        )
        return int(round(self.max_cluster_size_um3 / voxel_volume))

    @cached_property
    def ball_xy_size(self) -> int:
        voxel_sizes = self.voxel_sizes
        mean_in_plane_pixel_size = 0.5 * (
            float(voxel_sizes[2]) + float(voxel_sizes[1])
        )
        return int(round(self.ball_xy_size_um / mean_in_plane_pixel_size))

    @cached_property
    def ball_z_size(self) -> int:
        voxel_sizes = self.voxel_sizes
        ball_z_size = int(round(self.ball_z_size_um / float(voxel_sizes[0])))

        if not ball_z_size:
            raise ValueError(
                "Ball z size has been calculated to be 0 voxels."
                " This may be due to large axial spacing of your data or the "
                "ball_z_size_um parameter being too small. "
                "Please check input parameters are correct. "
                "Note that cellfinder requires high resolution data in all "
                "dimensions, so that cells can be detected in multiple "
                "image planes."
            )
        return ball_z_size

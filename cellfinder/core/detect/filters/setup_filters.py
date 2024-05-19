from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Tuple, Type

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
    Configuration class with all the parameters used during 2d and 3d filtering
    and structure splitting.
    """

    plane_original_np_dtype: Type[np.number] = np.uint16
    """
    The numpy data type of the input data that will be passed to the filtering
    pipeline.
    """

    plane_shape: Tuple[int, int] = (1, 1)
    """
    The shape of each plane of the input data as (height, width) - i.e.
    (axis 1, axis 2) in the z-stack where z is the first axis.
    """

    start_plane: int = 0
    """The index of first plane to process, in the input data (inclusive)."""
    end_plane: int = 1
    """
    The index of the last plane at which to stop processing the input data
    (not inclusive).
    """

    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """
    Tuple of voxel sizes in each dimension (z, y, x). We use this to convert
    from um to pixel sizes.
    """
    soma_spread_factor: float = 1.4
    """Spread factor for soma size - how much it may stretch in the images."""
    soma_diameter_um: float = 16
    """Diameter of a typical soma in um."""
    max_cluster_size_um3: float = 100_000
    """
    Maximum size of a cluster (bright area) that will be processed, in um.
    """
    ball_xy_size_um: float = 6
    """
    Diameter of the 3d spherical kernel filter in the x/y dimensions in um.
    See `ball_xy_size` for size in pixels.
    """
    ball_z_size_um: float = 15
    """
    Diameter of the 3d spherical kernel filter in the z dimension in um.
    See `ball_z_size` for size in pixels.

    `ball_z_size` also determines to the minimum number of planes that are
    stacked before can filter the central plane of the stack.
    """

    ball_overlap_fraction: float = 0.6
    """
    Fraction of overlap between the bright area and the spherical kernel,
    to be considered a single ball.
    """
    log_sigma_size: float = 0.2
    """Size of the sigma for the Gaussian filter."""
    n_sds_above_mean_thresh: float = 10
    """
    Number of standard deviations above the mean to use for a threshold to
    define bright areas.
    """

    outlier_keep: bool = False
    """Whether to keep outlier structures during detection."""
    artifact_keep: bool = False
    """Whether to keep artifact structures during detection."""

    save_planes: bool = False
    """Whether to save the planes during detection."""
    plane_directory: Optional[str] = None
    """Directory path where to save the planes, if saving."""

    batch_size: int = 1
    """
    The number of planes to process in each batch of the 2d/3d filters.

    For CPU, each plane in a batch is 2d filtered (the slowest filters) in its
    own sub-process. But 3d filtering happens in a single thread. So larger
    batches will use more processes but can speed up filtering until IO/3d
    filters become the bottleneck.

    For CUDA, 2d and 3d filtering happens on the GPU and the larger the batch
    size, the better the performance. Until it fills up the GPU memory - after
    which it becomes slower.

    In all cases, higher batch size means more RAM used.
    """

    num_prefetch_batches: int = 2
    """
    The number of batches to load into memory.

    This many batches are loaded in memory so the next batch is ready to be
    sent to the filters as soon as the previous batch is done.

    The higher the number the more RAM used, but it can also speed up
    processing if IO becomes a limiting factor.
    """

    torch_device: str = "cpu"
    """
    The device on which to run the computation.

    Either "cpu" or PyTorch's GPU device name, such as "cuda" to run on the
    first GPU.
    """

    n_free_cpus: int = 2
    """
    Number of free CPU cores to keep available and not use during parallel
    processing.
    """

    n_splitting_iter: int = 10
    """
    During the structure splitting phase we iteratively shrink the bright areas
    and re-filter with the 3d filter. This is the number of iterations to do.

    This is a maximum because we also stop if there are no more structures left
    during any iteration.
    """

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
            self.plane_original_np_dtype, self.filtering_dtype
        )

    @cached_property
    def filtering_dtype(self) -> Type[np.floating]:
        original_dtype = self.plane_original_np_dtype
        original_max_int = get_max_possible_int_value(original_dtype)

        # does original data fit in float32
        if original_max_int <= get_max_possible_int_value(np.float32):
            return np.float32
        # what about float64
        if original_max_int <= get_max_possible_int_value(np.float64):
            return np.float64
        raise TypeError("Input array data type is too big for a float64")

    @cached_property
    def detection_dtype(self) -> Type[np.unsignedinteger]:
        # filtering data type is signed. But, filtering should only produce
        # values >= 0. So unsigned is safe
        working_dtype = self.filtering_dtype
        # filtering type is at most float64. So it'll always fit in uint64
        # also
        max_int = get_max_possible_int_value(working_dtype)
        if max_int <= get_max_possible_int_value(np.uint32):
            return np.uint32
        assert max_int <= get_max_possible_int_value(
            np.uint64
        ), "filtering_dtype should be at most float64"
        return np.uint64

    @cached_property
    def clipping_value(self) -> int:
        return get_max_possible_int_value(self.filtering_dtype) - 2

    @cached_property
    def threshold_value(self) -> int:
        return get_max_possible_int_value(self.filtering_dtype) - 1

    @cached_property
    def soma_centre_value(self) -> int:
        return get_max_possible_int_value(self.filtering_dtype)

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
    def n_planes(self) -> int:
        return self.end_plane - self.start_plane

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

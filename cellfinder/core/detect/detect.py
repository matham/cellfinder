"""
Detection is run in three steps:

1. 2D filtering
2. 3D filtering
3. Structure detection

In steps 1. and 2. filters are applied, and any bright points detected
post-filter are marked. To avoid using a separate mask array to mark the
bright points, the input data is clipped to [0, (max_val - 2)]
(max_val is the maximum value that the image data type can store), and:
- (max_val - 1) is used to mark bright points during 2D filtering
- (max_val) is used to mark bright points during 3D filtering
"""

import dataclasses
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from brainglobe_utils.cells.cells import Cell

from cellfinder.core import logger, types
from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.volume_filter import VolumeFilter


def main(
    signal_array: types.array,
    start_plane: int,
    end_plane: int,
    voxel_sizes: Tuple[float, float, float],
    soma_diameter: float,
    max_cluster_size: float,
    ball_xy_size: float,
    ball_z_size: float,
    ball_overlap_fraction: float,
    soma_spread_factor: float,
    n_free_cpus: int,
    log_sigma_size: float,
    n_sds_above_mean_thresh: float,
    outlier_keep: bool = False,
    artifact_keep: bool = False,
    save_planes: bool = False,
    plane_directory: Optional[str] = None,
    batch_size: int = 1,
    torch_device: str = "cpu",
    split_ball_xy_size: int = 3,
    split_ball_z_size: int = 3,
    split_ball_overlap_fraction: float = 0.8,
    split_soma_diameter: int = 7,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """
    Perform cell candidate detection on a 3D signal array.

    Parameters
    ----------
    signal_array : numpy.ndarray
        3D array representing the signal data.

    start_plane : int
        Index of the starting plane for detection.

    end_plane : int
        Index of the ending plane for detection.

    voxel_sizes : Tuple[float, float, float]
        Tuple of voxel sizes in each dimension (z, y, x).

    soma_diameter : float
        Diameter of the soma in physical units.

    max_cluster_size : float
        Maximum size of a cluster in physical units.

    ball_xy_size : float
        Size of the XY ball used for filtering in physical units.

    ball_z_size : float
        Size of the Z ball used for filtering in physical units.

    ball_overlap_fraction : float
        Fraction of overlap allowed between balls.

    soma_spread_factor : float
        Spread factor for soma size.

    n_free_cpus : int
        Number of free CPU cores available for parallel processing.

    log_sigma_size : float
        Size of the sigma for the log filter.

    n_sds_above_mean_thresh : float
        Number of standard deviations above the mean threshold.

    outlier_keep : bool, optional
        Whether to keep outliers during detection. Defaults to False.

    artifact_keep : bool, optional
        Whether to keep artifacts during detection. Defaults to False.

    save_planes : bool, optional
        Whether to save the planes during detection. Defaults to False.

    plane_directory : str, optional
        Directory path to save the planes. Defaults to None.

    batch_size : int, optional
        The number of planes to process in each batch. Defaults to 1.
        For CPU, there's no benifit for a larger batch size. Only a memory
        usage increase. For CUDA, the larger the batch size the better the
        performance. Until it fills up the GPU memory - after which it
        becomes slower.

    torch_device : str, optional
        The device on which to run the computation. By default it's "cpu".
        To run on a gpu, specifiy the PyTorch device name, such as "cuda" to
        run on the first GPU.

    callback : Callable[int], optional
        A callback function that is called every time a plane has finished
        being processed. Called with the plane number that has finished.

    Returns
    -------
    List[Cell]
        List of detected cells.
    """
    start_time = datetime.now()

    if not np.issubdtype(signal_array.dtype, np.number):
        raise ValueError(
            "signal_array must be a numpy datatype, but has datatype "
            f"{signal_array.dtype}"
        )

    if signal_array.ndim != 3:
        raise ValueError("Input data must be 3D")

    if end_plane == -1:
        end_plane = len(signal_array)
    n_planes = max(min(len(signal_array), end_plane) - start_plane, 0)

    # pytorch requires floats for many operations, we can use float64,
    # but it's slower
    filterting_dtype = "float32"

    settings = DetectionSettings(
        plane_shape=signal_array.shape[1:],
        plane_original_np_dtype=signal_array.dtype,
        filterting_dtype=filterting_dtype,
        voxel_sizes=voxel_sizes,
        soma_spread_factor=soma_spread_factor,
        soma_diameter_um=soma_diameter,
        max_cluster_size_um3=max_cluster_size,
        ball_xy_size_um=ball_xy_size,
        ball_z_size_um=ball_z_size,
        start_plane=start_plane,
        end_plane=end_plane,
        n_planes=n_planes,
        n_free_cpus=n_free_cpus,
        ball_overlap_fraction=ball_overlap_fraction,
        log_sigma_size=log_sigma_size,
        n_sds_above_mean_thresh=n_sds_above_mean_thresh,
        outlier_keep=outlier_keep,
        artifact_keep=artifact_keep,
        save_planes=save_planes,
        plane_directory=plane_directory,
        batch_size=batch_size,
        torch_device=torch_device,
    )

    # replicate the settings specific to splitting
    splitting_settings = DetectionSettings(**dataclasses.asdict(settings))
    splitting_settings.ball_z_size = split_ball_z_size
    splitting_settings.ball_xy_size = split_ball_xy_size
    splitting_settings.ball_overlap_fraction = split_ball_overlap_fraction
    splitting_settings.soma_diameter = split_soma_diameter
    # always run on cpu because copying to gpu overhead is likely slower than
    # any benfit for detection on smallish volumes
    splitting_settings.torch_device = "cpu"

    # Create 3D analysis filter
    mp_3d_filter = VolumeFilter(settings=settings)

    # Create 2D analysis filter
    mp_tile_processor = TileProcessor(
        plane_shape=settings.plane_shape,
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        n_sds_above_mean_thresh=settings.n_sds_above_mean_thresh,
        log_sigma_size=settings.log_sigma_size,
        soma_diameter=settings.soma_diameter,
        torch_device=settings.torch_device,
        dtype=settings.filterting_dtype,
    )

    with torch.inference_mode(True):
        # process the data
        mp_3d_filter.process(
            mp_tile_processor, signal_array, callback=callback
        )
        cells = mp_3d_filter.get_results(splitting_settings)

    time_elapsed = datetime.now() - start_time
    s = f"Detection complete. Found {len(cells)} cells in {time_elapsed}"
    logger.debug(s)
    print(s)
    return cells

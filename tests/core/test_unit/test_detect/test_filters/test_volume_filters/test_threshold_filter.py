import numpy as np
import pytest
import torch

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.threshold_filter import (
    ThresholdFilter3D,
)


def get_filtered_data(
    data: np.ndarray,
    soma_diameter=16,
    log_sigma_size=0.2,
    n_sds_above_mean_thresh=10.0,
    n_sds_above_mean_tiled_thresh=10.0,
    tiled_thresh_tile_size_xy=0.0,
    tiled_thresh_tile_size_z=1.0,
) -> np.ndarray:
    settings = DetectionSettings(plane_original_np_dtype=np.uint16)
    data = data.astype(settings.filtering_dtype)
    z, y, x = data.shape

    tile_processor = TileProcessor(
        plane_shape=data.shape[1:],
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=soma_diameter,
        log_sigma_size=log_sigma_size,
        n_sds_above_mean_thresh=n_sds_above_mean_thresh,
        torch_device="cpu",
        dtype=settings.filtering_dtype.__name__,
        use_scipy=True,
    )

    filtered, _, enhanced = tile_processor.get_tile_mask(
        torch.from_numpy(data)
    )

    if tiled_thresh_tile_size_xy:
        tf = ThresholdFilter3D(
            plane_height=y,
            plane_width=x,
            tile_xy_size=int(round(tiled_thresh_tile_size_xy * soma_diameter)),
            tile_z_size=int(round(tiled_thresh_tile_size_z * soma_diameter)),
            n_sds_above_mean_thresh=n_sds_above_mean_tiled_thresh,
            threshold_value=settings.threshold_value,
            dtype=settings.filtering_dtype.__name__,
            batch_size=1,
            torch_device="cpu",
        )
        tf.append(enhanced, filtered)

        res = []
        if tf.ready:
            # get done data
            tf.walk()
            res = tf.get_processed_planes()
            res = [r[None, ...] for r in res]

        if tf.flush():
            # and any newly flushed data
            tf.walk()
            res2 = tf.get_processed_planes()
            res.extend([r[None, ...] for r in res2])

        filtered = torch.cat(res, dim=0)

    is_threshold = (filtered == settings.threshold_value).numpy()
    assert is_threshold.shape == data.shape

    return is_threshold


def test_2d_filter_tiled_threshold_2_spots():
    # create 2 bright areas of 5x5 = 25px, one bright, one darker
    data = np.zeros((1, 50, 50))
    data[0, 3:8, 3:8] = 5
    data[0, 43:48, 43:48] = 20

    # medium plane threshold should get only very bright area
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=2,
        tiled_thresh_tile_size_xy=0,
    )
    assert 20 <= np.sum(filtered) <= 35

    # with small tiles (size of soma) the mean would be high for the tiles with
    # both bright areas so we should get no pixels
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=2,
        tiled_thresh_tile_size_xy=1,
    )
    assert not np.sum(filtered)

    # but with a very low tiled threshold we should get same as with plane
    # threshold only
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=-2,
        tiled_thresh_tile_size_xy=1,
    )
    assert 20 <= np.sum(filtered) <= 35

    # and with a low plane threshold as well we should get everything
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=0,
        n_sds_above_mean_tiled_thresh=-2,
        tiled_thresh_tile_size_xy=1,
    )
    assert 40 <= np.sum(filtered) <= 60


def test_3d_filter_tiled_threshold_large_spot_tile_vs_no_tile(
    synthetic_single_spot_large,
):
    data, _, _ = synthetic_single_spot_large

    # medium plane threshold, no tile, should get only very bright area
    filtered_untiled = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.3,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=2,
        tiled_thresh_tile_size_xy=0,
    )
    untiled_sum = np.sum(filtered_untiled, axis=(1, 2))
    assert np.sum(filtered_untiled) > 7 * 7 * 7  # radius is ~6

    # with a very low tiled threshold we should get same as with plane
    # threshold only
    filtered_low_tiled = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.3,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=-2,
        tiled_thresh_tile_size_xy=1,
        tiled_thresh_tile_size_z=1,
    )
    low_tiled_sum = np.sum(filtered_low_tiled, axis=(1, 2))
    # compare voxel counts for each z-plane
    assert np.sum(np.abs(untiled_sum - low_tiled_sum)) < 10

    # and with a low plane threshold as well we should get even more
    filtered_low_thresh = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.3,
        n_sds_above_mean_thresh=0,
        n_sds_above_mean_tiled_thresh=-2,
        tiled_thresh_tile_size_xy=1,
        tiled_thresh_tile_size_z=1,
    )
    assert np.sum(filtered_untiled) < 2 * np.sum(filtered_low_thresh)


def test_3d_filter_tiled_threshold_large_spot_tile_size(
    synthetic_single_spot_smooth,
):
    """
    This shows that when tiling in xy, if we tile in z we get less voxels above
    threshold at the edges, than if we don't tile in z (i.e. each plane is
    processed independently).
    """
    data, _, _ = synthetic_single_spot_smooth

    filtered_tile_med_z = get_filtered_data(
        data,
        soma_diameter=6,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=0,
        n_sds_above_mean_tiled_thresh=4,
        tiled_thresh_tile_size_xy=3,
        tiled_thresh_tile_size_z=3,
    )
    print(np.sum(filtered_tile_med_z, axis=(1, 2)))
    med_non_zero = np.sum(np.sum(filtered_tile_med_z, axis=(1, 2)) > 0)

    filtered_tile_small_z = get_filtered_data(
        data,
        soma_diameter=6,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=0,
        n_sds_above_mean_tiled_thresh=4,
        tiled_thresh_tile_size_xy=3,
        tiled_thresh_tile_size_z=1 / 7,
    )
    print(np.sum(filtered_tile_small_z, axis=(1, 2)))
    small_non_zero = np.sum(np.sum(filtered_tile_small_z, axis=(1, 2)) > 0)

    assert small_non_zero > med_non_zero + 2


@pytest.mark.parametrize(
    "shape", [(1, 50, 23), (1, 23, 50), (1, 25, 25), (1, 57, 57)]
)
def test_2d_filter_tiled_threshold_odd_shapes(shape):
    # our tile size is 5 * 5 = 25, check that plane shapes that don't fit two
    # tiles or are not multiple of tile size still works
    # create bright area of 5x5 = 25px
    data = np.zeros(shape)
    data[0, 3:8, 3:8] = 5

    # use tiles size of 25 (5 x soma diameter of 5)
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=1,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size_xy=5,
    )
    # about 25 pixels should be marked
    assert 20 <= np.sum(filtered) <= 30


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_2d_filter_tiled_threshold_odd_tile_size(size):
    # check that tiny tile sizes works.
    data = np.zeros((1, 10, 10))

    # use tiles size of 25 (5 x soma diameter of 5)
    filtered = get_filtered_data(
        data,
        soma_diameter=1,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=1,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size_xy=size,
    )
    assert filtered.shape == (1, 10, 10)

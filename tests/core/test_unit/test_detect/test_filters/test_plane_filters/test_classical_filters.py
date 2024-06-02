import numpy as np
import pytest
import torch

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.plane.classical_filter import PeakEnhancer
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.tools.IO import read_with_dask
from cellfinder.core.tools.tools import inference_wrapper


@pytest.fixture
def filtered_data_array(repo_data_path):
    # loads an input and already 2d filtered data set
    data_path = (
        repo_data_path
        / "integration"
        / "detection"
        / "structure_split_test"
        / "signal"
    )
    filtered_path = (
        repo_data_path / "integration" / "detection" / "filter" / "2d_filter"
    )
    tiles_path = (
        repo_data_path / "integration" / "detection" / "filter" / "tiles"
    )
    return (
        read_with_dask(str(data_path)),
        read_with_dask(str(filtered_path)),
        read_with_dask(str(tiles_path)),
    )


@pytest.mark.parametrize(
    "torch_device,use_scipy", [("cpu", False), ("cpu", True), ("cuda", False)]
)
@inference_wrapper
def test_2d_filtering(filtered_data_array, torch_device, use_scipy):
    # test that the 2d plane filtering (median, gauss, laplacian) matches
    if torch_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    # check input data size/type is as expected
    data, filtered, tiles = filtered_data_array
    data = np.asarray(data)
    filtered = np.asarray(filtered)
    tiles = np.asarray(tiles)
    assert data.dtype == np.uint16
    assert filtered.dtype == np.uint16
    assert data.shape == filtered.shape

    settings = DetectionSettings(plane_original_np_dtype=np.uint16)
    # convert to working type and send to cpu/cuda
    data = torch.from_numpy(settings.filter_data_converter_func(data))
    data = data.to(torch_device)

    tile_processor = TileProcessor(
        plane_shape=data[0, :, :].shape,
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=16,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=10,
        torch_device=torch_device,
        dtype=settings.filtering_dtype.__name__,
        use_scipy=use_scipy,
    )

    # apply filter and get data back
    filtered_our, tiles_our = tile_processor.get_tile_mask(data)
    filtered_our = filtered_our.cpu().numpy().astype(np.uint16)
    tiles_our = tiles_our.cpu().numpy()

    assert filtered_our.shape == filtered.shape
    # the number of pixels per plane that are different
    diff = np.sum(np.sum(filtered_our != filtered, axis=2), axis=1)
    # total pixels per plane
    n_pixels = data.shape[1] * data.shape[2]
    # fraction pixels that are different
    frac = diff / n_pixels
    # 99.999% same
    assert np.all(np.less(frac, 1 - 0.99_999))

    assert tiles_our.shape == tiles.shape
    assert tiles_our.dtype == tiles.dtype
    assert np.array_equal(tiles_our, tiles)


@pytest.mark.parametrize(
    "plane_size",
    [(1, 2), (2, 1), (2, 2), (2, 3), (3, 3), (2, 5), (22, 33), (200, 200)],
)
@inference_wrapper
def test_2d_filter_padding(plane_size):
    # check that filter padding works correctly for different sized inputs -
    # even if the input is smaller than filter sizes
    settings = DetectionSettings(plane_original_np_dtype=np.uint16)
    data = np.random.randint(0, 500, size=(1, *plane_size))
    data = data.astype(settings.filtering_dtype)

    tile_processor = TileProcessor(
        plane_shape=plane_size,
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=16,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=10,
        torch_device="cpu",
        dtype=settings.filtering_dtype.__name__,
        use_scipy=False,
    )

    filtered, _ = tile_processor.get_tile_mask(torch.from_numpy(data))
    filtered = filtered.numpy()
    assert filtered.shape == data.shape


@inference_wrapper
def test_even_filter_kernel():
    with pytest.raises(ValueError):
        try:
            n = PeakEnhancer.median_filter_size
            PeakEnhancer.median_filter_size = 4
            PeakEnhancer(
                "cpu",
                torch.float32,
                clipping_value=5,
                laplace_gaussian_sigma=3.0,
                use_scipy=False,
            )
        finally:
            PeakEnhancer.median_filter_size = n

    enhancer = PeakEnhancer(
        "cpu",
        torch.float32,
        clipping_value=5,
        laplace_gaussian_sigma=3.0,
        use_scipy=False,
    )

    assert enhancer.gaussian_filter_size % 2

    _, _, x, y = enhancer.lap_kernel.shape
    assert x % 2, "Should be odd"
    assert y % 2, "Should be odd"
    assert x == y


@pytest.mark.parametrize(
    "sizes",
    [((1, 1), (1, 1)), ((1, 2), (1, 1)), ((2, 1), (1, 1)), ((22, 33), (3, 4))],
)
@inference_wrapper
def test_tile_walker_size(sizes, soma_diameter=5):
    plane_size, tile_size = sizes
    walker = TileWalker(plane_size, soma_diameter=soma_diameter)
    assert walker.tile_height == 10
    assert walker.tile_width == 10

    data = torch.rand((1, *plane_size), dtype=torch.float32)
    tiles = walker.get_bright_tiles(data)
    assert tiles.shape == (1, *tile_size)

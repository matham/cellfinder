import numpy as np
import pytest
import torch

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.tools.IO import read_with_dask


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

    with torch.inference_mode(True):
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
    # 99.99% same
    assert np.all(np.less(frac, 1 - 0.9999))

    assert tiles_our.shape == tiles.shape
    assert tiles_our.dtype == tiles.dtype
    assert np.array_equal(tiles_our, tiles)

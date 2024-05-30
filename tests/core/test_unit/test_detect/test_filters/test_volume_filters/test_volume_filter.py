import numpy as np
import pytest
import torch
from pytest_mock.plugin import MockerFixture

from cellfinder.core.detect.detect import main
from cellfinder.core.tools.IO import read_with_dask
from cellfinder.core.tools.threading import ExecutionFailure

# even though we are testing volume filter as unit test, we are running through
# main and mocking VolumeFilter because that's the easiest way to instantiate
# it with proper args and run it


class ExceptionTest(Exception):
    pass


@pytest.fixture
def filtered_data_array(repo_data_path):
    # loads an input and already 3d filtered data set
    data_path = (
        repo_data_path
        / "integration"
        / "detection"
        / "structure_split_test"
        / "signal"
    )
    filtered_path = (
        repo_data_path / "integration" / "detection" / "filter" / "3d_filter"
    )
    return (
        read_with_dask(str(data_path)),
        read_with_dask(str(filtered_path)),
    )


def raise_exception(*args, **kwargs):
    raise ExceptionTest("Bad times")


def run_main_assert_exception():
    # run volume filter - it should raise the ExceptionTest via
    # ExecutionFailure from thread/process
    try:
        # must be on cpu b/c only on cpu do we do 2d filtering in subprocess
        # lots of planes so it doesn't end naturally quickly
        main(signal_array=np.zeros((500, 500, 500)), torch_device="cpu")
        assert False, "should have raised exception"
    except ExecutionFailure as e:
        e2 = e.__cause__
        assert type(e2) is ExceptionTest and e2.args == ("Bad times",)


def test_2d_filter_process_exception(mocker: MockerFixture):
    # check sub-process that does 2d filter. That exception ends things clean
    mocker.patch(
        "cellfinder.core.detect.filters.volume.volume_filter._plane_filter",
        new=raise_exception,
    )
    run_main_assert_exception()


def test_2d_filter_feeder_thread_exception(mocker: MockerFixture):
    # check data feeder thread. That exception ends things clean
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    mocker.patch.object(
        VolumeFilter, "_feed_signal_batches", new=raise_exception
    )
    run_main_assert_exception()


def test_2d_filter_cell_detection_thread_exception(mocker: MockerFixture):
    # check cell detection thread. That exception ends things clean
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    mocker.patch.object(
        VolumeFilter, "_run_filter_thread", new=raise_exception
    )
    run_main_assert_exception()


def test_3d_filter_main_thread_exception(mocker: MockerFixture):
    # raises exception in the _process method in the main thread - after the
    # subprocess and secondary threads were spun up. This makes sure that those
    # subprocess and threads don't get stuck if main thread crashes
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    mocker.patch.object(VolumeFilter, "_process", new=raise_exception)
    with pytest.raises(ExceptionTest):
        main(signal_array=np.zeros((500, 500, 500)), torch_device="cpu")


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_feeder_thread_batch(batch_size: int):
    # checks various batch sizes to see if there are issues
    # this also tests a batch size of 3 but 5 planes. So the feeder thread
    # will feed us a batch of 3 and a batch of 2. It tests that filters can
    # handle unequal batch sizes
    planes = []

    def callback(z):
        planes.append(z)

    main(
        signal_array=np.zeros((5, 50, 50)),
        torch_device="cpu",
        batch_size=batch_size,
        callback=callback,
    )

    assert planes == list(range(1, 4))


def test_not_enough_planes():
    # checks that even if there are not enough planes for volume filtering, it
    # doesn't raise errors or gets stuck
    planes = []

    def callback(z):
        planes.append(z)

    main(
        signal_array=np.zeros((2, 50, 50)),
        torch_device="cpu",
        callback=callback,
    )

    assert not planes


def test_filtered_plane_range(mocker: MockerFixture):
    # check that even if input data is negative, filtered data is non-negative
    detector = mocker.patch(
        "cellfinder.core.detect.filters.volume.volume_filter.CellDetector",
        autospec=True,
    )

    # input data in range (-500, 500)
    data = ((np.random.random((6, 50, 50)) - 0.5) * 1000).astype(np.float32)
    data[1:3, 25:30, 25:30] = 5000
    main(signal_array=data)

    calls = detector.return_value.process.call_args_list
    assert len(calls)
    for call in calls:
        plane, *_ = call.args
        # should have either zero or soma value or both
        assert len(np.unique(plane)) in (1, 2)
        assert np.min(plane) >= 0


def test_saving_filtered_planes(tmp_path):
    # check that we can save filtered planes
    path = tmp_path / "save_planes"
    path.mkdir()

    main(
        signal_array=np.zeros((6, 50, 50)),
        save_planes=True,
        plane_directory=str(path),
    )

    files = [p.name for p in path.iterdir() if p.is_file()]
    # we're skipping first and last plane that isn't filtered due to kernel
    assert len(files) == 4
    assert files == [
        "plane_0002.tif",
        "plane_0003.tif",
        "plane_0004.tif",
        "plane_0005.tif",
    ]


def test_saving_filtered_planes_no_dir():
    # asked to save but didn't provide directory
    with pytest.raises(ExecutionFailure) as exc_info:
        main(
            signal_array=np.zeros((6, 50, 50)),
            save_planes=True,
            plane_directory=None,
        )
    assert type(exc_info.value.__cause__) is ValueError


@pytest.mark.parametrize(
    "torch_device,use_scipy", [("cpu", False), ("cpu", True), ("cuda", False)]
)
def test_3d_filtering_saved(
    filtered_data_array, torch_device, use_scipy, no_free_cpus, tmp_path
):
    # test that the full 2d/3d matches the saved data
    if torch_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    # check input data size/type is as expected
    data, filtered = filtered_data_array
    data = np.asarray(data)
    filtered = np.asarray(filtered)
    assert data.dtype == np.uint16
    assert filtered.dtype == np.uint32
    assert data.shape == (filtered.shape[0] + 2, *filtered.shape[1:])

    path = tmp_path / "3d_filter"
    path.mkdir()
    main(
        signal_array=data,
        voxel_sizes=(5, 2, 2),
        soma_diameter=16,
        max_cluster_size=100000,
        ball_xy_size=6,
        ball_z_size=15,
        ball_overlap_fraction=0.6,
        soma_spread_factor=1.4,
        n_free_cpus=no_free_cpus,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=10,
        save_planes=True,
        plane_directory=str(path),
    )

    filtered_our = np.asarray(read_with_dask(str(path)))
    assert filtered_our.shape == filtered.shape
    assert filtered_our.dtype == np.uint16

    # the number of pixels per plane that are different
    diff = np.sum(np.sum(filtered_our != filtered, axis=2), axis=1)
    # total pixels per plane
    n_pixels = data.shape[1] * data.shape[2]
    # fraction pixels that are different
    frac = diff / n_pixels
    # 99.9% same
    assert np.all(np.less(frac, 1 - 0.999))

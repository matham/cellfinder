import numpy as np
import pytest
from pytest_mock.plugin import MockerFixture

from cellfinder.core.detect.detect import main
from cellfinder.core.tools.threading import ExecutionFailure

# even though we are testing volume filter as unit test, we are running through
# main and mocking VolumeFilter because that's the easiest way to instantiate
# it with proper args and run it


class ExceptionTest(Exception):
    pass


def raise_exception(*args, **kwargs):
    raise ExceptionTest("Bad times")


def call_main(**kwargs):
    d = {
        "start_plane": 0,
        "end_plane": 1,
        "voxel_sizes": (5, 5, 5),
        "soma_diameter": 30,
        "max_cluster_size": 100_000,
        "ball_xy_size": 6,
        "ball_z_size": 15,
        "ball_overlap_fraction": 0.5,
        "soma_spread_factor": 1.4,
        "n_free_cpus": 1,
        "log_sigma_size": 0.2,
        "n_sds_above_mean_thresh": 10,
    }
    d.update(kwargs)
    main(**d)


def run_main_assert_exception():
    # run volume filter - it should raise the ExceptionTest via
    # ExecutionFailure from thread/process
    try:
        # must be on cpu b/c only on cpu do we do 2d filtering in subprocess
        call_main(signal_array=np.zeros((4, 50, 50)), torch_device="cpu")
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
        call_main(signal_array=np.zeros((5, 50, 50)), torch_device="cpu")


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_feeder_thread_batch(batch_size: int):
    # checks various batch sizes to see if there are issues
    # this also tests a batch size of 3 but 4 planes. So the feeder thread
    # will feed us a batch of 3 and a batch of 1. It tests that filters can
    # handle unequal batch sizes
    call_main(
        signal_array=np.zeros((4, 50, 50)),
        torch_device="cpu",
        batch_size=batch_size,
    )


def test_filtered_plane_range(mocker: MockerFixture):
    # check that even if input data is negative, filtered data is positive
    detector = mocker.patch(
        "cellfinder.core.detect.filters.volume.volume_filter.CellDetector",
        autospec=True,
    )

    # input data in range (-500, 500)
    data = ((np.random.random((6, 50, 50)) - 0.5) * 1000).astype(np.float32)
    call_main(signal_array=data)

    calls = detector.return_value.process.call_args_list
    assert len(calls)
    for call in calls:
        plane, *_ = call.args
        # should have at least background and foreground pixels
        assert len(np.unique(plane)) >= 2
        assert np.min(plane) >= 0

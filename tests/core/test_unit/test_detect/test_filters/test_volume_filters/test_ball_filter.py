import pytest

from cellfinder.core.detect.filters.volume.ball_filter import BallFilter


def test_filter_not_read():
    bf = BallFilter(
        plane_height=50,
        plane_width=50,
        ball_xy_size=3,
        ball_z_size=3,
        overlap_fraction=0.5,
        threshold_value=1,
        soma_centre_value=1,
        tile_height=10,
        tile_width=10,
        dtype="float32",
    )
    assert not bf.ready

    with pytest.raises(TypeError):
        bf.get_processed_planes()

    with pytest.raises(TypeError):
        bf.walk()

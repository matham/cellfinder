import pytest
import torch

from cellfinder.core.detect.filters.volume.ball_filter import BallFilter

bf_kwargs = {
    "plane_height": 50,
    "plane_width": 50,
    "ball_xy_size": 3,
    "ball_z_size": 3,
    "overlap_fraction": 0.5,
    "threshold_value": 1,
    "soma_centre_value": 1,
    "tile_height": 10,
    "tile_width": 10,
    "dtype": "float32",
    "torch_device": "cpu",
}


def test_filter_not_ready():
    bf = BallFilter(**bf_kwargs)
    assert not bf.ready

    with pytest.raises(TypeError):
        bf.get_processed_planes()

    with pytest.raises(TypeError):
        bf.walk()


@pytest.mark.parametrize(
    "sizes", [(1, 0, 0), (2, 1, 0), (3, 1, 1), (4, 2, 1), (5, 2, 2), (6, 3, 2)]
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_filter_plane_params(sizes, batch_size):
    kernel_size, first_plane, remaining_planes = sizes
    # we get exactly one plane out of a volume that is the same size as the
    # kernel first_plane is index of first valid plane. Plus remaining is last
    # index. Plus 1 is size
    assert kernel_size == first_plane + 1 + remaining_planes

    kwargs = bf_kwargs.copy()
    kwargs["ball_z_size"] = kernel_size
    bf = BallFilter(use_mask=True, **kwargs)

    assert not bf.ready
    assert not bf.volume.shape[0]
    assert not bf.inside_brain_tiles.shape[0]

    planes = torch.zeros(
        (batch_size, bf_kwargs["plane_height"], bf_kwargs["plane_width"]),
        device=bf_kwargs["torch_device"],
        dtype=bf.volume.dtype,
    )
    masks = torch.ones(
        (batch_size, *bf.inside_brain_tiles.shape[1:]),
        device=bf_kwargs["torch_device"],
        dtype=bf.inside_brain_tiles.dtype,
    )

    bf.append(planes, masks)
    assert bf.volume.shape[0] == first_plane + batch_size
    assert bf.inside_brain_tiles.shape[0] == first_plane + batch_size

    n_gotten = 0
    if bf.ready:
        assert remaining_planes - (batch_size - 1) <= 0

        processed = bf.get_processed_planes()
        assert processed.shape[1:] == planes.shape[1:]
        n_gotten = processed.shape[0]
    else:
        assert remaining_planes - (batch_size - 1) > 0

    if bf.flush():
        assert remaining_planes

        assert bf.volume.shape[0] >= kernel_size
        assert bf.inside_brain_tiles.shape[0] >= kernel_size

        processed = bf.get_processed_planes()
        assert processed.shape[1:] == planes.shape[1:]
        n_gotten += processed.shape[0]
    else:
        assert not remaining_planes

    assert bf.ready
    assert n_gotten == batch_size


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 5])
def test_filtered_planes(kernel_size, batch_size):
    kwargs = bf_kwargs.copy()
    kwargs["ball_z_size"] = kernel_size
    bf = BallFilter(**kwargs, use_mask=False)

    data = torch.empty(
        (batch_size, kwargs["plane_height"], kwargs["plane_width"]),
        dtype=getattr(torch, kwargs["dtype"]),
        device=kwargs["torch_device"],
    )

    num_planes = 20
    sent_planes = 0
    gotten_planes = 0
    num_padded_planes = kernel_size - (kernel_size // 2 + 1)

    for _ in range(num_planes // batch_size):
        bf.append(data)
        sent_planes += batch_size
        # volume only includes batch and some padding from end of last batch
        assert bf.volume.shape[0] <= batch_size + kernel_size - 1

        if bf.ready:
            # no need to walk because walking only modifies the contents not
            # size of volume
            planes = bf.get_processed_planes()
            gotten_planes += planes.shape[0]

    assert gotten_planes == sent_planes - num_padded_planes

    if gotten_planes < sent_planes:
        assert bf.flush()

        planes = bf.get_processed_planes()
        gotten_planes += planes.shape[0]
        assert gotten_planes == sent_planes
    else:
        assert not bf.flush()

import numpy as np
import pytest
import torch

from cellfinder.core.detect.filters.volume.ball_filter import (
    BallFilter,
    get_kernel,
)

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


@pytest.mark.parametrize("xy_size", list(range(1, 7)))
@pytest.mark.parametrize("z_size", list(range(1, 7)))
def test_kernel_symetry(xy_size, z_size):
    kernel = get_kernel(xy_size, z_size)
    for axis in range(3):
        flipped = np.flip(kernel, axis=axis)
        assert np.allclose(flipped, kernel)

    assert kernel.shape[0] == xy_size
    assert kernel.shape[1] == xy_size
    assert kernel.shape[2] == z_size


def test_filter_not_ready():
    bf = BallFilter(**bf_kwargs)
    assert not bf.ready

    with pytest.raises(TypeError):
        bf.get_processed_planes()

    with pytest.raises(TypeError):
        bf.get_raw_planes()

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
    assert not bf.planes
    assert not bf.inside_brain_tiles.shape[0]

    planes = torch.zeros(
        (batch_size, bf_kwargs["plane_height"], bf_kwargs["plane_width"]),
        device=bf_kwargs["torch_device"],
        dtype=getattr(torch, bf_kwargs["dtype"]),
    )
    masks = torch.ones(
        (batch_size, bf.n_vertical_tiles, bf.n_horizontal_tiles),
        device=bf_kwargs["torch_device"],
        dtype=torch.bool,
    )

    bf.append(planes, masks)
    assert len(bf.planes) == first_plane + batch_size
    assert len(bf.inside_brain_tiles) == first_plane + batch_size

    n_gotten = 0
    if bf.ready:
        assert remaining_planes - (batch_size - 1) <= 0

        processed = bf.get_processed_planes()
        for p in processed:
            assert p.shape == planes.shape[1:]
        n_gotten = len(processed)
    else:
        assert remaining_planes - (batch_size - 1) > 0

    if bf.flush():
        assert remaining_planes

        assert len(bf.planes) >= kernel_size
        assert len(bf.inside_brain_tiles) >= kernel_size

        processed = bf.get_processed_planes()
        for p in processed:
            assert p.shape == planes.shape[1:]
        n_gotten += len(processed)
    else:
        assert not remaining_planes

    assert bf.ready
    assert n_gotten == batch_size


@pytest.mark.parametrize(
    "sizes", [(1, 0, 0), (2, 0, 1), (3, 1, 1), (4, 1, 2), (5, 2, 2), (6, 2, 3)]
)
def test_filter_padding(sizes):
    # checks that for a given kernel size, the start / end padding matches as
    # expected. The start padding is always the lessor (when even)
    kernel_size, *padding = sizes

    assert BallFilter.min_xy_padding(kernel_size) == tuple(padding)
    assert BallFilter.min_z_padding(kernel_size) == tuple(padding)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 5])
def test_filtered_planes(kernel_size, batch_size):
    kwargs = bf_kwargs.copy()
    kwargs["ball_z_size"] = kernel_size
    bf = BallFilter(**kwargs, use_mask=False)

    num_planes = 20
    n_batches = num_planes // batch_size
    total_planes = n_batches * batch_size
    sent_planes = 0
    gotten_planes = 0
    num_padded_planes = kernel_size - (kernel_size // 2 + 1)

    h, w = kwargs["plane_height"], kwargs["plane_width"]
    data = torch.arange(total_planes * h * w).reshape((total_planes, h, w))
    data = data.to(
        dtype=getattr(torch, kwargs["dtype"]), device=kwargs["torch_device"]
    )
    data_np = data.numpy()

    all_raw_planes = []
    for i in range(n_batches):
        bf.append(
            data[i * batch_size : (i + 1) * batch_size],
            raw_planes=data_np[i * batch_size : (i + 1) * batch_size],
        )
        sent_planes += batch_size
        # volume only includes batch and some padding from end of last batch
        assert len(bf.planes) <= batch_size + kernel_size - 1

        if bf.ready:
            # no need to walk because walking only modifies the contents not
            # size of volume
            planes = bf.get_processed_planes()
            raw_planes = bf.get_raw_planes()
            all_raw_planes.extend(raw_planes)

            assert len(raw_planes) == len(planes)
            for raw_plane in raw_planes:
                assert raw_plane.shape == planes[0].shape

            gotten_planes += len(planes)

    assert gotten_planes == sent_planes - num_padded_planes

    if gotten_planes < sent_planes:
        assert bf.flush()

        planes = bf.get_processed_planes()
        raw_planes = bf.get_raw_planes()
        all_raw_planes.extend(raw_planes)

        assert len(raw_planes) == len(planes)
        for raw_plane in raw_planes:
            assert raw_plane.shape == planes[0].shape

        gotten_planes += len(planes)
        assert gotten_planes == sent_planes
        assert len(all_raw_planes) == sent_planes
    else:
        assert not bf.flush()

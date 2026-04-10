import numpy as np

from cellfinder.core.detect.filters.volume.laplacian_filter import (
    get_5_stencil_2d,
    get_7_stencil,
    get_27_stencil,
)


def test_5_stencil_2d():
    res = get_5_stencil_2d((1, 1, 1))

    # from wiki and previous 2d laplacian kernel, when voxel size is 1 in all
    # axes and 2d it should be as follows
    p1 = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    p2 = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]
    p3 = p1

    kernel = np.array([p1, p2, p3])
    assert np.allclose(kernel, res)


def test_7_stencil():
    res = get_7_stencil((1, 1, 1))

    # from wiki, when voxel size is 1 in all axes it should be as follows
    p1 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    p2 = [
        [0, 1, 0],
        [1, -6, 1],
        [0, 1, 0],
    ]
    p3 = p1

    kernel = np.array([p1, p2, p3])
    assert np.allclose(kernel, res)


def test_27_stencil():
    res = get_27_stencil((1, 1, 1))

    # from wiki, when voxel size is 1 in all axes it should be as follows
    p1 = [
        [2, 3, 2],
        [3, 6, 3],
        [2, 3, 2],
    ]
    p2 = [
        [3, 6, 3],
        [6, -88, 6],
        [3, 6, 3],
    ]
    p3 = p1

    kernel = np.array([p1, p2, p3]) / 26
    assert np.allclose(kernel, res)

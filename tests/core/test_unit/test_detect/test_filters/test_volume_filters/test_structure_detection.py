from typing import Dict, List, Tuple

import numpy as np
import pytest

from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    Point,
    get_non_zero_dtype_min,
    get_structure_centre,
)


def coords_to_points(
    coords_arrays: Dict[int, np.ndarray]
) -> Dict[int, List[Point]]:
    # Convert from arrays to dicts
    coords = {}
    for sid in coords_arrays:
        coords[sid] = []
        for row in coords_arrays[sid]:
            coords[sid].append(Point(row[0], row[1], row[2]))
    return coords


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.uint64, 2**64 - 1),
        (np.uint32, 2**32 - 1),
        (np.uint16, 2**16 - 1),
        (np.uint8, 2**8 - 1),
        (np.float32, 2**23),  # mantissa determine whole max int representable
        (np.float64, 2**52),
    ],
)
def test_get_non_zero_dtype_min(dtype: np.dtype, expected: int) -> None:
    assert get_non_zero_dtype_min(np.arange(10, dtype=dtype)) == 1
    assert get_non_zero_dtype_min(np.zeros(10, dtype=dtype)) == expected


@pytest.fixture()
def three_d_cross() -> np.ndarray:
    return np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )


@pytest.fixture()
def structure(three_d_cross: np.ndarray) -> np.ndarray:
    coords = np.array(np.where(three_d_cross)).transpose()
    return coords


def test_get_structure_centre(structure: np.ndarray) -> None:
    result_point = get_structure_centre(structure)
    assert (result_point[0], result_point[1], result_point[2]) == (
        1,
        1,
        1,
    )


# Image volume dimensions
width, height = 3, 2
depth = 2

# Each item in the test data contains:
#
# - A list of indices to mark as structure pixels (ordering: [z, y, x])
# - A dict of expected structure coordinates
test_data = [
    (
        # No pixels
        [],
        {},
    ),
    (
        # Two pixels connected in a single structure along x
        [(0, 0, 0), (0, 0, 1)],
        {1: [Point(0, 0, 0), Point(1, 0, 0)]},
    ),
    (
        # Two pixels connected in a single structure along y
        [(0, 0, 0), (0, 1, 0)],
        {1: [Point(0, 0, 0), Point(0, 1, 0)]},
    ),
    (
        # Two pixels connected in a single structure along z
        [(0, 0, 0), (1, 0, 0)],
        {1: [Point(0, 0, 0), Point(0, 0, 1)]},
    ),
    (
        # Four pixels all connected and spread across x-y-z
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)],
        {1: [Point(0, 0, 0), Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1)]},
    ),
    (
        # three initially disconnected pixels that then get merged
        # by a fourth pixel
        [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        {
            1: [
                Point(1, 1, 0),
                Point(1, 0, 1),
                Point(0, 1, 1),
                Point(1, 1, 1),
            ]
        },
    ),
    (
        # Three pixels in x-y plane that require structure merging
        [(0, 0, 1), (1, 0, 0), (1, 0, 1)],
        {
            1: [
                Point(1, 0, 0),
                Point(0, 0, 1),
                Point(1, 0, 1),
            ]
        },
    ),
    (
        # Two disconnected single-pixel structures
        [(0, 0, 0), (0, 0, 2)],
        {1: [Point(0, 0, 0)], 2: [Point(2, 0, 0)]},
    ),
    (
        # Two disconnected single-pixel structures along a diagonal
        [(0, 0, 0), (1, 1, 1)],
        {1: [Point(0, 0, 0)], 2: [Point(1, 1, 1)]},
    ),
]


# Due to  https://github.com/numba/numba/issues/9576 we need to run np.uint64
# before smaller sizes
# we don't officially support float in cell detection, but it still works
@pytest.mark.parametrize(
    "dtype",
    [np.uint64, np.uint8, np.uint16, np.uint32, np.float32, np.float64],
)
@pytest.mark.parametrize("pixels,expected_coords", test_data)
def test_detection(
    dtype: np.dtype,
    pixels: List[Tuple[int, int, int]],
    expected_coords: Dict[int, List[Point]],
) -> None:
    # original dtype is the dtype of the original data. filtering_dtype
    # is the data type used during ball filtering. Currently, it can be at most
    # float64. So original dtype cannot support 64-bits because it won't fit in
    # float64.
    # detection_dtype is the type that must be used during detection to fit the
    # values used during filtering (i.e. settings.soma_centre_value)
    settings = DetectionSettings(plane_original_np_dtype=dtype)

    if dtype == np.uint64:
        with pytest.raises(TypeError):
            # should raise error for uint64 - too big for float64
            filtering_dtype = settings.filtering_dtype
            # do something with it so linter doesn't complain
            assert filtering_dtype
        return

    data = np.zeros((depth, height, width)).astype(settings.detection_dtype)
    detector = CellDetector(height, width, 0, 0)
    # similar to numba issue #9576 we can't pass to init a large value once
    # a 32 bit type was used for detector. So pass it with custom method
    detector._set_soma(settings.soma_centre_value)

    # This is the value used by BallFilter to mark pixels
    for pix in pixels:
        data[pix] = settings.soma_centre_value

    previous_plane = None
    for plane in data:
        previous_plane = detector.process(plane, previous_plane)

    coords = detector.get_structures()
    assert coords_to_points(coords) == expected_coords

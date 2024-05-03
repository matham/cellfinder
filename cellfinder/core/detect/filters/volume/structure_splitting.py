from typing import List, Tuple

import numpy as np
import torch

from cellfinder.core import logger
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter_cuda import BallFilter
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)


class StructureSplitException(Exception):
    pass


def get_shape(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> List[int]:
    # +1 because difference. TEST:
    shape = [int((dim.max() - dim.min()) + 1) for dim in (xs, ys, zs)]
    return shape


def coords_to_volume(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    ball_radius: int,
    dtype: np.dtype,
    threshold_value: int,
) -> np.ndarray:
    ball_diameter = ball_radius * 2
    # Expanded to ensure the ball fits even at the border
    expanded_shape = [
        dim_size + ball_diameter for dim_size in get_shape(zs, xs, ys)
    ]
    volume = torch.zeros(expanded_shape, dtype=dtype)

    x_min, y_min, z_min = xs.min(), ys.min(), zs.min()

    relative_xs = np.array((xs - x_min + ball_radius), dtype=np.int64)
    relative_ys = np.array((ys - y_min + ball_radius), dtype=np.int64)
    relative_zs = np.array((zs - z_min + ball_radius), dtype=np.int64)

    # OPTIMISE: vectorize
    for rel_x, rel_y, rel_z in zip(relative_xs, relative_ys, relative_zs):
        volume[rel_z, rel_x, rel_y] = threshold_value
    return volume


def ball_filter_imgs(
    volume: np.ndarray, settings: DetectionSettings
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ball filtering to a 3D volume and detect cell centres.

    Uses the `BallFilter` class to perform ball filtering on the volume
    and the `CellDetector` class to detect cell centres.

    Args:
        volume (np.ndarray): The 3D volume to be filtered.
        threshold_value (int): The threshold value for ball filtering.
        soma_centre_value (int): The value representing the soma centre.
        ball_xy_size (int, optional):
            The size of the ball filter in the XY plane. Defaults to 3.
        ball_z_size (int, optional):
            The size of the ball filter in the Z plane. Defaults to 3.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the filtered volume and the cell centres.

    """
    # OPTIMISE: reuse ball filter instance

    good_tiles_mask = torch.ones((volume.shape[0], 1, 1), dtype=torch.bool)

    plane_width, plane_height = volume.shape[1:]
    detection_dtype = settings.detection_dtype
    batch_size = settings.batch_size

    bf = BallFilter(settings=settings)
    start_z = bf.first_valid_plane
    cell_detector = CellDetector(
        plane_width,
        plane_height,
        start_z=start_z,
        soma_centre_value=settings.soma_centre_value,
    )

    previous_plane = None
    for z in range(0, volume.shape[0], batch_size):
        item = volume[z : z + batch_size, :, :]

        bf.append(item, good_tiles_mask[z : z + batch_size, :, :])
        if bf.ready:
            bf.walk()
            middle_planes = bf.get_middle_planes()
            volume[z : z + middle_planes.shape[0], :, :] = torch.from_numpy(
                middle_planes
            )

            middle_planes = middle_planes.astype(detection_dtype)
            for plane in middle_planes:
                previous_plane = cell_detector.process(plane, previous_plane)
    return cell_detector.get_cell_centres()


def iterative_ball_filter(
    volume: np.ndarray, settings: DetectionSettings
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Apply iterative ball filtering to the given volume.
    The volume is eroded at each iteration, by subtracting 1 from the volume.

    Parameters:
        volume (np.ndarray): The input volume.

    Returns:
        Tuple[List[int], List[np.ndarray]]: A tuple containing two lists:
            The structures found in each iteration.
            The cell centres found in each iteration.
    """
    ns = []
    centres = []

    for i in range(settings.n_splitting_iter):
        cell_centres = ball_filter_imgs(volume, settings)
        volume.sub_(1)

        n_structures = len(cell_centres)
        ns.append(n_structures)
        centres.append(cell_centres)
        if n_structures == 0:
            break

    return ns, centres


def check_centre_in_cuboid(centre: np.ndarray, max_coords: np.ndarray) -> bool:
    """
    Checks whether a coordinate is in a cuboid
    :param centre: x,y,z coordinate
    :param max_coords: far corner of cuboid
    :return: True if within cuboid, otherwise False
    """
    relative_coords = centre
    if (relative_coords > max_coords).all():
        logger.info(
            'Relative coordinates "{}" exceed maximum volume '
            'dimension of "{}"'.format(relative_coords, max_coords)
        )
        return False
    else:
        return True


def split_cells(
    cell_points: np.ndarray, settings: DetectionSettings
) -> np.ndarray:
    """
    Split the given cell points into individual cell centres.

    Args:
        cell_points (np.ndarray): Array of cell points with shape (N, 3),
            where N is the number of cell points and each point is represented
            by its x, y, and z coordinates.
        outlier_keep (bool, optional): Flag indicating whether to keep outliers
            during the splitting process. Defaults to False.

    Returns:
        np.ndarray: Array of absolute cell centres with shape (M, 3),
            where M is the number of individual cells and each centre is
            represented by its x, y, and z coordinates.
    """
    orig_centre = get_structure_centre(cell_points)

    xs = cell_points[:, 0]
    ys = cell_points[:, 1]
    zs = cell_points[:, 2]

    orig_corner = np.array(
        [
            orig_centre[0] - (orig_centre[0] - xs.min()),
            orig_centre[1] - (orig_centre[1] - ys.min()),
            orig_centre[2] - (orig_centre[2] - zs.min()),
        ]
    )

    relative_orig_centre = np.array(
        [
            orig_centre[0] - orig_corner[0],
            orig_centre[1] - orig_corner[1],
            orig_centre[2] - orig_corner[2],
        ]
    )

    original_bounding_cuboid_shape = get_shape(xs, ys, zs)

    ball_radius = settings.ball_xy_size // 2
    dtype = getattr(torch, settings.filterting_dtype)
    vol = coords_to_volume(
        xs,
        ys,
        zs,
        ball_radius=ball_radius,
        dtype=dtype,
        threshold_value=settings.threshold_value,
    )

    settings.ball_z_size = 3
    settings.ball_xy_size = 3
    settings.ball_overlap_fraction = 0.8
    settings.plane_shape = vol.shape[1:]
    settings.soma_diameter = 7
    settings.soma_spread_factor = 1.4
    settings.max_cluster_size = 1348
    settings.start_plane = 0
    settings.end_plane = vol.shape[0]
    settings.n_planes = settings.end_plane
    settings.tile_dim1 = settings.plane_shape[0]
    settings.tile_dim2 = settings.plane_shape[1]

    # centres is a list of arrays of centres (1 array of centres per ball run)
    ns, centres = iterative_ball_filter(vol, settings)
    ns.insert(0, 1)
    centres.insert(0, np.array([relative_orig_centre]))

    best_iteration = ns.index(max(ns))

    # TODO: put constraint on minimum centres distance ?
    relative_centres = centres[best_iteration]

    if not settings.outlier_keep:
        # TODO: change to checking whether in original cluster shape
        original_max_coords = np.array(original_bounding_cuboid_shape)
        relative_centres = np.array(
            [
                x
                for x in relative_centres
                if check_centre_in_cuboid(x, original_max_coords)
            ]
        )

    absolute_centres = np.empty((len(relative_centres), 3))
    # FIXME: extract functionality
    absolute_centres[:, 0] = orig_corner[0] + relative_centres[:, 0]
    absolute_centres[:, 1] = orig_corner[1] + relative_centres[:, 1]
    absolute_centres[:, 2] = orig_corner[2] + relative_centres[:, 2]

    return absolute_centres

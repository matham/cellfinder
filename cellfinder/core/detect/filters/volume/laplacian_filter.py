import math
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# the kernels defined below are based on https://en.wikipedia.org/wiki/
# Discrete_Laplace_operator#Implementation_via_operator_discretization and
# O’Reilly, R. C., & Beck, J. M. (2006). A family of large-stencil discrete
# Laplacian approximations in three-dimensions. Int. J. Numer. Methods Eng,
# 1-16.


def get_5_stencil_2d(voxel_sizes: tuple[float, float, float]) -> np.ndarray:
    dz2, dy2, dx2 = [v**2 for v in voxel_sizes]
    # rows are first axis, i.e. y. Cols are x. pn are the z planes

    p2 = [
        [0, 1 / dy2, 0],
        [1 / dx2, -2 * (1 / dy2 + 1 / dx2), 1 / dx2],
        [0, 1 / dy2, 0],
    ]
    p3 = p1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    return np.array([p1, p2, p3])


def get_7_stencil(voxel_sizes: tuple[float, float, float]) -> np.ndarray:
    dz2, dy2, dx2 = [v**2 for v in voxel_sizes]
    # rows are first axis, i.e. y. Cols are x. pn are the z planes

    p1 = [
        [0, 0, 0],
        [0, 1 / dz2, 0],
        [0, 0, 0],
    ]
    p2 = [
        [0, 1 / dy2, 0],
        [1 / dx2, -2 * (1 / dz2 + 1 / dy2 + 1 / dx2), 1 / dx2],
        [0, 1 / dy2, 0],
    ]
    p3 = p1

    return np.array([p1, p2, p3])


def get_27_stencil(voxel_sizes: tuple[float, float, float]) -> np.ndarray:
    # first row of eq 22 is the overall formula. There are 6 faces, 12 edges,
    # 8 corners, and 1 center = 27 points. Normalization factors divide by the
    # dist squared because we're computing 2nd derivative (h2 in formula with
    # 1/2 or 1/3 factors for edge and corner respectively under isotropy).
    dz, dy, dx = voxel_sizes
    # face distance squared
    dz2, dy2, dx2 = [v**2 for v in voxel_sizes]
    # edge distance squared
    dxy2 = dx**2 + dy**2
    dxz2 = dx**2 + dz**2
    dyz2 = dy**2 + dz**2
    # corner distance squared
    dxyz2 = dx**2 + dy**2 + dz**2

    # rows are first axis, i.e. y. Cols are x. pn are the z planes.
    # Faces, edges, and corners simply are normalized by squared distance from
    # center point.
    # Center point is just the negative sum of all of them. Numerator should
    # sum to 26. In original formula, if h is 1 and dx/dy/dz is 1, this will be
    # 6 + 12 / 2 + 8 / 3, which is 44 / 3 as in the paper
    center = (
        2 / dx2
        + 2 / dy2
        + 2 / dz2
        + 4 / dxy2
        + 4 / dxz2
        + 4 / dyz2
        + 8 / dxyz2
    )
    p1 = [
        [1 / dxyz2, 1 / dyz2, 1 / dxyz2],
        [1 / dxz2, 1 / dz2, 1 / dxz2],
        [1 / dxyz2, 1 / dyz2, 1 / dxyz2],
    ]
    p2 = [
        [1 / dxy2, 1 / dy2, 1 / dxy2],
        [1 / dx2, -center, 1 / dx2],
        [1 / dxy2, 1 / dy2, 1 / dxy2],
    ]
    p3 = p1

    # overall normalization factor
    arr = np.array([p1, p2, p3]) * 3 / 13
    return arr


class LaplacianFilter3D:

    num_batches_before_ready: int
    """
    The number of batches of size `batch_size` passed to `append`
    before `ready` would return True.
    """

    kernel: torch.Tensor

    # the added planes. They are padded for conv by 1 on each side and are of
    # shape 1, 1, y, x
    planes: list[torch.Tensor] = []

    processed_planes: list[torch.Tensor] = []

    def __init__(
        self,
        voxel_sizes: tuple[float, float, float],
        dtype: str,
        batch_size: int = 1,
        torch_device: str = "cpu",
        filter_type: Literal["2d", "3d_faces", "3d_full"] = "2d",
    ):
        self.num_batches_before_ready = int(math.ceil(3 / batch_size))
        self.planes = []
        self.processed_planes = []
        # Index of the middle plane in the min # of planes needed for filtering
        self.middle_z_idx = 1

        self.side_data: list[Sequence[torch.Tensor | np.ndarray]] = []

        match filter_type:
            case "2d":
                kernel = get_5_stencil_2d(voxel_sizes)
            case "3d_faces":
                kernel = get_7_stencil(voxel_sizes)
            case "3d_full":
                kernel = get_27_stencil(voxel_sizes)
            case _:
                raise ValueError(f"Unknown filter volume type {filter_type}")

        kernel = torch.tensor(
            kernel.tolist(),
            dtype=getattr(torch, dtype),
            device=torch_device,
            pin_memory=torch_device != "cpu",
        )
        # we defined a negative center kernel, like scipy defaults. But then we
        # need to flip sign after using it so that positive peaks are positive,
        # instead multiply by -1 now
        self.kernel = kernel * -1

    @property
    def ready(self) -> bool:
        """
        Return whether enough planes have been appended to run the filter
        using `walk`.
        """
        return len(self.planes) >= 3

    def _append(self, planes: list[torch.Tensor]) -> int:
        """
        Append new data to volume and remove previously processed volume data.
        """
        remaining_start = 0
        if self.planes:
            if len(self.planes) < 3:
                # we haven't processed any yet
                remaining_start = 0
            else:
                # if we had 3+ planes, we previously processed this many. E.g.
                # with 3 planes, we processed only one plane before
                remaining_start = len(self.planes) - 2
                del self.planes[:remaining_start]

            self.planes.extend(planes)
        else:
            # need to pad with 1 start plane before middle plane
            self.planes = [planes[0].clone()] + list(planes)

        return remaining_start

    def append(
        self,
        planes: torch.Tensor,
        *side_data: torch.Tensor | np.ndarray,
    ) -> None:
        """
        Add a new z-stack to the filter.

        Previous stacks passed to `append` are removed, except enough planes
        at the top of the previous z-stack to provide padding so we can filter
        starting from the first plane in `planes`. The first time we call
        `append`, `first_valid_plane` is the first plane to actually be
        filtered in the z-stack due to lack of padding.

        So make sure to call `walk`/`get_processed_planes` before calling
        `append` again.

        Parameters
        ----------
        planes : torch.Tensor
            The z-stack. There can be one or more planes in the stack, but it
            must have 3 dimensions. Input data is not modified.
        """
        for sdata in side_data:
            if planes.shape[0] != sdata.shape[0]:
                raise ValueError(
                    "Side data must have same number of planes as data"
                )
        # convert to list of 2d planes
        planes = list(planes)
        # we use a 3x3x3 kernel. Each plane must be padded by 1 on all sides
        planes = [
            F.pad(p[None, None, ...], (1, 1, 1, 1), "replicate")
            for p in planes
        ]
        remaining_start = self._append(planes)

        if remaining_start:
            del self.side_data[:remaining_start]
            del self.processed_planes[:remaining_start]

        # convert to list of if size z, with tuples of 2d xy planes
        self.side_data.extend(zip(*side_data))

    def flush(self) -> bool:
        """
        Ensure to get unprocessed planes, because this calls append.
        """
        if not self.planes:
            return False

        # we just need one plane of padding at the end
        remaining_start = self._append([self.planes[-1].clone()])

        if remaining_start:
            del self.side_data[:remaining_start]
            del self.processed_planes[:remaining_start]

        return True

    def get_processed_planes(self) -> list[torch.Tensor]:
        """
        After passing enough planes to `append`, and after `walk`, this returns
        a copy of the processed planes as a numpy z-stack.

        It only starts returning planes corresponding to plane
        `first_valid_plane` relative to the first planes passed to `append`.
        E.g. if `ball_z_size` is 3 and `first_valid_plane` is 1, and you passed
        5 planes total to `append`, then this will have returned planes [1, 3].

        Notice the last plane was not included, because we return only "middle"
        planes - planes that can correspond to the center of a ball.
        """
        if not self.ready:
            raise TypeError("Not enough planes were appended")

        num_processed = len(self.planes) - 2
        assert num_processed
        assert num_processed == len(self.processed_planes)

        return self.processed_planes[:]

    def get_processed_side_data_planes(
        self,
    ) -> list[list[torch.Tensor | np.ndarray]]:
        if not self.ready:
            raise TypeError("Not enough planes were appended")

        num_processed = len(self.planes) - 2
        assert num_processed

        side_data = [list(s) for s in zip(*self.side_data[:num_processed])]
        return side_data

    def walk(self) -> None:
        """
        Applies the filter to all the planes passed to `append`.

        May only be called if `ready` was True.

        You can get the processed planes from `get_processed_planes`.
        """
        if not self.ready:
            raise TypeError("Called walk before enough planes were appended")

        self.processed_planes.extend(_filter_planes(self.planes, self.kernel))


@torch.jit.script
def _filter_planes(
    planes: list[torch.Tensor],
    kernel: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Sets each plane (in-place) to threshold_value, where the corresponding
    enhanced_plane > mean + n_sds_above_mean_thresh*std. Each plane will be
    set to zero elsewhere.
    """
    # add batch, channel dims and split into the 3 kernel planes
    kp0 = kernel[None, None, 0, ...]
    kp1 = kernel[None, None, 1, ...]
    kp2 = kernel[None, None, 2, ...]

    processed_planes = []
    # we are guaranteed to have a padding plane at start and end of planes
    for i in range(1, len(planes) - 1):
        # add batch, channel dims. They are already padded by 1 on all sides
        # todo: consider doing it as a batch
        p0 = F.conv2d(planes[i - 1], kp0, padding="valid")
        p1 = F.conv2d(planes[i], kp1, padding="valid")
        p2 = F.conv2d(planes[i + 1], kp2, padding="valid")

        processed_planes.append((p0 + p1 + p2)[0, 0, ...])

    return processed_planes

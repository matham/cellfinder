import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cellfinder.core.tools.image_processing import batch_tiled_mean_std


class ThresholdFilter3D:
    """
    A 3D ball filter.

    This runs a spherical kernel across the 2d planar dimensions
    of a *ball_z_size* stack of planes, and marks pixels in the middle
    plane of the stack that have a high enough intensity over the
    the spherical kernel.

    Parameters
    ----------
    plane_height, plane_width : int
        Height/width of the planes.
    threshold_value : int
        Value above which an individual pixel is considered to have
        a high intensity.
    dtype : str
        The data-type of the input planes and the type to use internally.
        E.g. "float32".
    batch_size: int
        The number of planes that will be typically passed in a single batch to
        `append`. This is only used to calculate `num_batches_before_ready`.
        Defaults to 1.
    torch_device: str
        The device on which the data and processing occurs on. Can be e.g.
        "cpu", "cuda" etc. Defaults to "cpu". Any data passed to the filter
        must be on this device. Returned data will also be on this device.
    """

    num_batches_before_ready: int
    """
    The number of batches of size `batch_size` passed to `append`
    before `ready` would return True.
    """

    tiled_mean_var: list[tuple[torch.Tensor, torch.Tensor]]

    def __init__(
        self,
        plane_height: int,
        plane_width: int,
        tile_xy_size: int,
        tile_z_size: int,
        n_sds_above_mean_thresh: float,
        threshold_value: int,
        dtype: str,
        batch_size: int = 1,
        torch_device: str = "cpu",
    ):
        # we do 50% overlap of tiles so there's no jumps at boundaries
        stride_xy = tile_xy_size // 2
        # make tile even for ease of computation
        tile_xy_size = stride_xy * 2

        # Due to 50% overlap, to get tiles we move the tile by half tile
        # (stride). Total moves will be y // stride - 2 (we start already
        # with mask on first tile). So add back 1 for the first tile. Partial
        # tiles are dropped
        n_y_tiles = max(plane_height // stride_xy - 1, 1) if stride_xy else 1
        n_x_tiles = max(plane_width // stride_xy - 1, 1) if stride_xy else 1
        do_tile_y = n_y_tiles >= 2
        do_tile_x = n_x_tiles >= 2

        # num edge pixels dropped b/c moving by stride would move tile off edge
        self.y_rem = plane_height % stride_xy
        self.x_rem = plane_width % stride_xy

        self.stride_xy = stride_xy
        self.tile_xy_size = tile_xy_size
        self.n_x_tiles = n_x_tiles
        self.n_y_tiles = n_y_tiles
        self.do_tile_x = do_tile_x
        self.do_tile_y = do_tile_y
        # we want at least one axis to have at least two tiles
        self.has_tiles = tile_xy_size >= 2 and (do_tile_y or do_tile_x)

        self.plane_height = plane_height
        self.plane_width = plane_width
        self.tile_z_size = tile_z_size
        self.n_sds_above_mean_thresh = n_sds_above_mean_thresh
        self.threshold_value = threshold_value

        self.num_batches_before_ready = int(
            math.ceil(self.tile_z_size / batch_size)
        )
        # Index of the middle plane in the planes
        self.middle_z_idx = int(np.floor(self.tile_z_size / 2))

        # Stores the current planes that are being filtered
        self.planes = []
        self.tiled_mean_var = []
        self.marked_planes: list[torch.Tensor] = []
        self.side_data: list[Sequence[torch.Tensor | np.ndarray]] = []

    @property
    def ready(self) -> bool:
        """
        Return whether enough planes have been appended to run the filter
        using `walk`.
        """
        return len(self.planes) >= self.tile_z_size

    def _append_to_planes(
        self,
        planes: Sequence[torch.Tensor],
        tiled_mean_var: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        """
        Append new data to volume and remove previously processed volume data.
        """
        remaining_start = 0
        if len(self.planes):
            if len(self.planes) < self.tile_z_size:
                num_remaining_with_padding = len(self.planes)
            else:
                num_remaining = self.tile_z_size - (self.middle_z_idx + 1)
                num_remaining_with_padding = num_remaining + self.middle_z_idx
            remaining_start = len(self.planes) - num_remaining_with_padding

            del self.tiled_mean_var[:remaining_start]
            self.tiled_mean_var.extend(tiled_mean_var)

            del self.planes[:remaining_start]
            self.planes.extend(planes)
        elif self.middle_z_idx > 0:
            # need to pad the start before middle plane
            self.tiled_mean_var = [
                tiled_mean_var[0] for _ in range(self.middle_z_idx)
            ]
            self.tiled_mean_var.extend(tiled_mean_var)

            self.planes = [planes[0] for _ in range(self.middle_z_idx)]
            self.planes.extend(planes)
        else:
            self.tiled_mean_var = list(tiled_mean_var)
            self.planes = list(planes)

        return remaining_start

    def append(
        self,
        data_planes: torch.Tensor | Sequence[torch.Tensor],
        marked_planes: torch.Tensor | Sequence[torch.Tensor],
        *side_data: torch.Tensor
        | np.ndarray
        | Sequence[torch.Tensor]
        | Sequence[np.ndarray],
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
        data_planes : torch.Tensor
            The z-stack. There can be one or more planes in the stack, but it
            must have 3 dimensions. Input data is not modified.
        """
        if len(data_planes) != len(marked_planes):
            raise ValueError(
                "Both data and marked planes must have same shape"
            )
        for sdata in side_data:
            if len(data_planes) != len(sdata):
                raise ValueError(
                    "Side data must have same number of planes as data"
                )

        # num edge pixels dropped b/c moving by stride would move tile off edge
        # todo: handle when no tiling
        tiled_mean_var = []
        for plane in data_planes:
            if self.do_tile_y:
                plane = plane[self.y_rem // 2 :, :]
            if self.do_tile_x:
                plane = plane[:, self.x_rem // 2 :]

            # add empty channel dim after z "batch" dim -> zcyx
            plane = plane[None, None, :, :]
            # unfold -> 3 dim, z, M, L. M is tile area, L is number of tiles
            unfolded = F.unfold(
                plane,
                (
                    self.tile_xy_size if self.do_tile_y else self.plane_height,
                    self.tile_xy_size if self.do_tile_x else self.plane_width,
                ),
                stride=self.stride_xy,
            )
            var, mean = torch.var_mean(unfolded[0, :, :], dim=0, correction=0)

            tiled_mean_var.append((mean, var))

        remaining_start = self._append_to_planes(
            list(data_planes), tiled_mean_var
        )

        if remaining_start:
            del self.marked_planes[:remaining_start]
            del self.side_data[:remaining_start]

        self.marked_planes.extend(marked_planes)
        self.side_data.extend(zip(*side_data))

    def flush(self) -> bool:
        """
        Ensure to get unprocessed planes, because this calls append.
        """
        # we need this many at end to process last plane
        pad = self.tile_z_size - self.middle_z_idx - 1

        if not len(self.planes) or not pad:
            return False

        self._append_to_planes(
            [self.planes[-1] for _ in range(pad)],
            [self.tiled_mean_var[-1] for _ in range(pad)],
        )

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

        num_processed = len(self.planes) - self.tile_z_size + 1
        assert num_processed

        return self.marked_planes[:num_processed]

    def get_processed_side_data_planes(
        self,
    ) -> list[list[torch.Tensor | np.ndarray]]:
        if not self.ready:
            raise TypeError("Not enough planes were appended")

        num_processed = len(self.planes) - self.tile_z_size + 1
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

        if not self.has_tiles:
            return

        _threshold_volume(
            self.planes,
            [p[0] for p in self.tiled_mean_var],
            [p[1] for p in self.tiled_mean_var],
            self.marked_planes,
            self.tile_xy_size,
            self.tile_z_size,
            self.x_rem,
            self.y_rem,
            self.n_x_tiles,
            self.n_y_tiles,
            self.do_tile_x,
            self.do_tile_y,
            self.stride_xy,
            self.middle_z_idx,
            self.n_sds_above_mean_thresh,
            self.threshold_value,
        )


@torch.jit.script
def _threshold_volume(
    data_planes: list[torch.Tensor],
    planes_mean: list[torch.Tensor],
    planes_var: list[torch.Tensor],
    marked_planes: list[torch.Tensor],
    tile_xy_size: int,
    tile_z_size: int,
    x_rem: int,
    y_rem: int,
    n_x_tiles: int,
    n_y_tiles: int,
    do_tile_x: bool,
    do_tile_y: bool,
    stride_xy: int,
    middle_z_idx: int,
    n_sds_above_mean_thresh: float,
    threshold_value: int,
) -> None:
    """
    Sets each plane (in-place) to threshold_value, where the corresponding
    enhanced_plane > mean + n_sds_above_mean_thresh*std. Each plane will be
    set to zero elsewhere.
    """
    num_process = len(data_planes) - tile_z_size + 1
    y, x = data_planes[0].shape
    tile_area = tile_xy_size if do_tile_x else 1
    if do_tile_y:
        tile_area *= tile_xy_size

    # we do a plane at a time, volume: i:i+num_z for plane i+middle. And unlike
    # xy where we stride at 50% tile size, for z we stride plane by plane
    for i in range(num_process):
        # average the tile areas, for each tile
        mean, std = batch_tiled_mean_std(
            planes_mean[i : i + tile_z_size],
            planes_var[i : i + tile_z_size],
            tile_xy_size * tile_xy_size,
        )
        threshold = mean + n_sds_above_mean_thresh * std

        # reshape it back into Y by X tiles, instead of YX being one dim
        threshold = threshold.reshape((n_y_tiles, n_x_tiles))

        # we need total size of n_tiles * stride + stride + rem for the
        # original size. So we add 2 strides and then chop off the excess above
        # rem. We center it because of 50% overlap, the first tile is actually
        # centered in between the first two strides
        offsets = [(0, y), (0, x)]
        for dim, do_tile, n_tiles, n, rem in [
            (0, do_tile_y, n_y_tiles, y, y_rem),
            (1, do_tile_x, n_x_tiles, x, x_rem),
        ]:
            if do_tile:
                repeats = (
                    torch.ones(
                        n_tiles, dtype=torch.int, device=data_planes[0].device
                    )
                    * stride_xy
                )
                # add total of 2 additional strides
                repeats[0] = 2 * stride_xy
                repeats[-1] = 2 * stride_xy
                output_size = (n_tiles + 2) * stride_xy

                threshold = threshold.repeat_interleave(
                    repeats, dim=dim, output_size=output_size
                )
                # drop the excess we gained from padding rem to whole stride
                offset = (stride_xy - rem) // 2
                offsets[dim] = offset, n + offset

        # can't use slice(...) objects in jit code so use actual indices
        (a, b), (c, d) = offsets
        threshold = threshold[a:b, c:d]
        # threshold is now the original plane size

        plane = marked_planes[i]
        above = torch.logical_and(
            plane == threshold_value,
            data_planes[middle_z_idx + i] > threshold,
        )

        t = torch.tensor(
            threshold_value, dtype=plane.dtype, device=plane.device
        )
        zero = torch.tensor(0, dtype=plane.dtype, device=plane.device)
        torch.where(above, t, zero, out=plane)

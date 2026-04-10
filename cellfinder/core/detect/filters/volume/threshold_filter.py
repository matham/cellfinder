import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


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
        self.tile_xy_size = tile_xy_size
        self.tile_z_size = tile_z_size
        self.n_sds_above_mean_thresh = n_sds_above_mean_thresh
        self.threshold_value = threshold_value

        self.num_batches_before_ready = int(
            math.ceil(self.tile_z_size / batch_size)
        )
        # Stores the current planes that are being filtered. Start with no data
        self.volume = torch.empty(
            (0, plane_height, plane_width),
            dtype=getattr(torch, dtype),
            device=torch_device,
        )
        # Index of the middle plane in the volume
        self.middle_z_idx = int(np.floor(self.tile_z_size / 2))

        self.marked_planes: list[torch.Tensor] = []
        self.side_data: list[Sequence[torch.Tensor | np.ndarray]] = []

    @property
    def ready(self) -> bool:
        """
        Return whether enough planes have been appended to run the filter
        using `walk`.
        """
        return self.volume.shape[0] >= self.tile_z_size

    def _append_to_volume(self, data_planes: torch.Tensor) -> int:
        """
        Append new data to volume and remove previously processed volume data.
        """
        remaining_start = 0
        if self.volume.shape[0]:
            if self.volume.shape[0] < self.tile_z_size:
                num_remaining_with_padding = self.volume.shape[0]
            else:
                num_remaining = self.tile_z_size - (self.middle_z_idx + 1)
                num_remaining_with_padding = num_remaining + self.middle_z_idx
            remaining_start = self.volume.shape[0] - num_remaining_with_padding

            self.volume = torch.cat(
                [self.volume[remaining_start:, :, :], data_planes],
                dim=0,
            )
        elif self.middle_z_idx > 0:
            # need to pad the start before middle plane
            self.volume = torch.cat(
                [
                    data_planes[:1],
                ]
                * self.middle_z_idx
                + [
                    data_planes,
                ],
                dim=0,
            )
        else:
            self.volume = data_planes.clone()

        return remaining_start

    def append(
        self,
        data_planes: torch.Tensor,
        marked_planes: torch.Tensor,
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
        data_planes : torch.Tensor
            The z-stack. There can be one or more planes in the stack, but it
            must have 3 dimensions. Input data is not modified.
        """
        if data_planes.shape != marked_planes.shape:
            raise ValueError(
                "Both data and marked planes must have same shape"
            )
        for sdata in side_data:
            if data_planes.shape[0] != sdata.shape[0]:
                raise ValueError(
                    "Side data must have same number of planes as data"
                )

        remaining_start = self._append_to_volume(data_planes)

        if remaining_start:
            del self.marked_planes[:remaining_start]
            del self.side_data[:remaining_start]

        for i in range(marked_planes.shape[0]):
            self.marked_planes.append(marked_planes[i, ...])

        self.side_data.extend(zip(*side_data))

    def flush(self) -> bool:
        """
        Ensure to get unprocessed planes, because this calls append.
        """
        # we need this many at end to process last plane
        pad = self.tile_z_size - self.middle_z_idx - 1

        if not self.volume.shape[0] or not pad:
            return False

        last_plane = self.volume[-1:, ...]
        data = torch.cat(
            [
                last_plane,
            ]
            * pad,
            dim=0,
        )
        self._append_to_volume(data)

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

        num_processed = self.volume.shape[0] - self.tile_z_size + 1
        assert num_processed

        return self.marked_planes[:num_processed]

    def get_processed_side_data_planes(
        self,
    ) -> list[list[torch.Tensor | np.ndarray]]:
        if not self.ready:
            raise TypeError("Not enough planes were appended")

        num_processed = self.volume.shape[0] - self.tile_z_size + 1
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

        _threshold_volume(
            self.marked_planes,
            self.volume,
            self.tile_xy_size,
            self.tile_z_size,
            self.middle_z_idx,
            self.n_sds_above_mean_thresh,
            self.threshold_value,
        )


@torch.jit.script
def _threshold_volume(
    marked_planes: list[torch.Tensor],
    data_planes: torch.Tensor,
    tile_xy_size: int,
    tile_z_size: int,
    middle_z_idx: int,
    n_sds_above_mean_thresh: float,
    threshold_value: int,
) -> None:
    """
    Sets each plane (in-place) to threshold_value, where the corresponding
    enhanced_plane > mean + n_sds_above_mean_thresh*std. Each plane will be
    set to zero elsewhere.
    """
    num_process = data_planes.shape[0] - tile_z_size + 1
    y, x = data_planes.shape[1:]

    # we do 50% overlap so there's no jumps at boundaries
    stride_xy = tile_xy_size // 2
    # make tile even for ease of computation
    tile_xy_size = stride_xy * 2
    # Due to 50% overlap, to get tiles we move the tile by half tile (stride).
    # Total moves will be y // stride - 2 (we start already with mask on first
    # tile). So add back 1 for the first tile. Partial tiles are dropped
    n_y_tiles = max(y // stride_xy - 1, 1) if stride_xy else 1
    n_x_tiles = max(x // stride_xy - 1, 1) if stride_xy else 1
    do_tile_y = n_y_tiles >= 2
    do_tile_x = n_x_tiles >= 2

    # we want at least one axis to have at least two tiles
    if tile_xy_size < 2 or (not do_tile_y) and (not do_tile_x):
        return

    # # in 2d filters each plane is independently normalized so absolute values
    # # mean nothing between planes. By normalizing to mean/std, assuming
    # # neighboring planes have somewhat similar stats, we can now tile across
    # # planes and use the same threshold across neighboring planes
    # std, mean = torch.std_mean(data_planes, dim=(1, 2), keepdim=True)
    # # don't edit in place since same plane may be "walked" in multiple calls
    # data_planes = data_planes - mean
    # # if min = max = zero, divide by 1 - it'll stay zero
    # std[std == 0] = 1
    # data_planes.div_(std)

    # num edge pixels dropped b/c moving by stride would move tile off edge
    y_rem = y % stride_xy
    x_rem = x % stride_xy
    data_planes_raw = data_planes
    if do_tile_y:
        data_planes = data_planes[:, y_rem // 2 :, :]
    if do_tile_x:
        data_planes = data_planes[:, :, x_rem // 2 :]

    # add empty channel dim after z "batch" dim -> zcyx
    data_planes = data_planes.unsqueeze(1)
    # unfold makes it 3 dim, z, M, L. L is number of tiles, M is tile area
    unfolded = F.unfold(
        data_planes,
        (tile_xy_size if do_tile_y else y, tile_xy_size if do_tile_x else x),
        stride=stride_xy,
    )

    # we do a plane at a time, volume: i:i+num_z for plane i+middle. And unlike
    # xy where we stride at 50% tile size, for z we stride plane by plane
    for i in range(num_process):
        # average the tile areas, for each tile
        std, mean = torch.std_mean(
            unfolded[i : i + tile_z_size, :, :], dim=(0, 1)
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
                        n_tiles, dtype=torch.int, device=data_planes.device
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
            data_planes_raw[middle_z_idx + i, :, :] > threshold,
        )

        t = torch.tensor(
            threshold_value, dtype=plane.dtype, device=plane.device
        )
        zero = torch.tensor(0, dtype=plane.dtype, device=plane.device)
        torch.where(above, t, zero, out=plane)

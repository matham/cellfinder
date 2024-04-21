import math
from typing import Tuple

import torch
import torch.nn.functional as F


class TileWalker:
    """
    A class to segment a 2D image into tiles, and mark each of the
    tiles as bright or dark depending on whether the average image
    value in each tile is above a threshold.

    The threshold is set using the tile of data containing the corner (0, 0).
    The mean and standard deviation of this tile is calculated, and
    the threshold set at 1 + mean + (2 * stddev).

    Attributes
    ----------
    bright_tiles_mask :
        An boolean array whose entries correspond to whether each tile is
        bright (1) or dark (0). The values are set in
        self.mark_bright_tiles().
    """

    def __init__(
        self, plane_shape: Tuple[int, int], soma_diameter: int
    ) -> None:
        self.img_width, self.img_height = plane_shape
        self.tile_width = soma_diameter * 2
        self.tile_height = soma_diameter * 2

        self.n_tiles_width = math.ceil(self.img_width / self.tile_width)
        self.n_tiles_height = math.ceil(self.img_height / self.tile_height)

    def get_bright_tiles(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Loop through tiles, and if the average value of a tile is
        greater than the intensity threshold mark the tile as bright
        in self.bright_tiles_mask.
        """
        return _get_bright_tiles(
            planes,
            self.n_tiles_width,
            self.n_tiles_height,
            self.tile_width,
            self.tile_height,
        )


@torch.jit.script
def _get_out_of_brain_threshold(
    planes: torch.Tensor, tile_width: int, tile_height: int
) -> torch.Tensor:
    corner_tiles = planes[:, 0:tile_width, 0:tile_height]
    corner_tiles = corner_tiles.reshape((planes.shape[0], -1))

    corner_intensity = torch.mean(corner_tiles, dim=1)
    corner_sd = torch.std(corner_tiles, dim=1)
    # add 1 to ensure not 0, as disables
    out_of_brain_thresholds = corner_intensity + 2 * corner_sd + 1

    # this is 1 dim, 1 value for each plane
    return out_of_brain_thresholds


@torch.jit.script
def _get_bright_tiles(
    planes: torch.Tensor,
    n_tiles_width: int,
    n_tiles_height: int,
    tile_width: int,
    tile_height: int,
) -> torch.Tensor:
    """
    Loop through tiles, and if the average value of a tile is
    greater than the intensity threshold mark the tile as bright
    in self.bright_tiles_mask.
    """
    bright_tiles_mask = torch.zeros(
        (planes.shape[0], n_tiles_width, n_tiles_height),
        dtype=torch.bool,
        device=planes.device,
    )
    out_of_brain_thresholds = _get_out_of_brain_threshold(
        planes, tile_width, tile_height
    )
    # Z -> ZXY
    thresholds = out_of_brain_thresholds.view(-1, 1, 1)

    tile_avg = F.avg_pool2d(
        planes.unsqueeze(1),  # ZXY -> ZCXY required for function
        (tile_width, tile_height),
        ceil_mode=False,
    )[
        :, 0, :, :
    ]  # ZCXY -> ZXY

    bright = tile_avg >= thresholds
    # in case the number of tiles is less than bright dim
    bright_tiles_mask[:, : bright.shape[1], : bright.shape[2]][bright] = True

    return bright_tiles_mask

import math

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

    def __init__(self, img: torch.Tensor, soma_diameter: int) -> None:
        self.img = img
        self.img_width, self.img_height = img.shape[1:]
        self.tile_width = soma_diameter * 2
        self.tile_height = soma_diameter * 2

        n_tiles_width = math.ceil(self.img_width / self.tile_width)
        n_tiles_height = math.ceil(self.img_height / self.tile_height)
        self.bright_tiles_mask = torch.zeros(
            (img.shape[0], n_tiles_width, n_tiles_height),
            dtype=torch.bool,
            device="cuda",
        )

        corner_tiles = img[
            :, 0 : self.tile_width, 0 : self.tile_height
        ].reshape((img.shape[0], -1))
        corner_intensity = torch.mean(corner_tiles, dim=1)
        corner_sd = torch.std(corner_tiles, dim=1)
        # add 1 to ensure not 0, as disables
        self.out_of_brain_thresholds = (corner_intensity + (2 * corner_sd)) + 1

    def mark_bright_tiles(self) -> None:
        """
        Loop through tiles, and if the average value of a tile is
        greater than the intensity threshold mark the tile as bright
        in self.bright_tiles_mask.
        """
        thresholds = self.out_of_brain_thresholds.unsqueeze(1).unsqueeze(2)

        bright = (
            F.avg_pool2d(
                self.img.unsqueeze(1),
                (self.tile_width, self.tile_height),
                ceil_mode=False,
            )[:, 0, :, :]
            >= thresholds
        )
        self.bright_tiles_mask[:, : bright.shape[1], : bright.shape[2]][
            bright
        ] = True

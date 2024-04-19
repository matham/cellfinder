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
        self.img_width, self.img_height = img.shape
        self.tile_width = soma_diameter * 2
        self.tile_height = soma_diameter * 2

        n_tiles_width = math.ceil(self.img_width / self.tile_width)
        n_tiles_height = math.ceil(self.img_height / self.tile_height)
        self.bright_tiles_mask = torch.zeros(
            (n_tiles_width, n_tiles_height), dtype=torch.bool, device="cuda"
        )

        corner_tile = img[0 : self.tile_width, 0 : self.tile_height]
        corner_intensity = torch.mean(corner_tile)
        corner_sd = torch.std(corner_tile)
        # add 1 to ensure not 0, as disables
        self.out_of_brain_threshold = (corner_intensity + (2 * corner_sd)) + 1

    def mark_bright_tiles(self) -> None:
        """
        Loop through tiles, and if the average value of a tile is
        greater than the intensity threshold mark the tile as bright
        in self.bright_tiles_mask.
        """
        threshold = self.out_of_brain_threshold
        if threshold == 0:
            return

        bright = (
            F.avg_pool2d(
                self.img.unsqueeze(0),
                (self.tile_width, self.tile_height),
                ceil_mode=False,
            )[0, :, :]
            >= threshold
        )
        self.bright_tiles_mask[: bright.shape[0], : bright.shape[1]] = True

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple

import numpy as np
import torch

from cellfinder.core import types
from cellfinder.core.detect.filters.plane.classical_filter import enhance_peaks
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker


@dataclass
class TileProcessor:
    """
    Attributes
    ----------
    clipping_value :
        Upper value that the input plane is clipped to.
    threshold_value :
        Value used to mark bright features in the input planes after they have
        been run through the 2D filter.
    """

    clipping_value: int
    threshold_value: int
    soma_diameter: int
    log_sigma_size: float
    n_sds_above_mean_thresh: float

    def get_tile_mask(
        self, planes: types.array, lock: Optional[Lock] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This thresholds the input plane, and returns a mask indicating which
        tiles are inside the brain.

        The input plane is:

        1. Clipped to [0, self.clipping_value]
        2. Run through a peak enhancement filter (see `classical_filter.py`)
        3. Thresholded. Any values that are larger than
           (mean + stddev * self.n_sds_above_mean_thresh) are set to
           self.threshold_value in-place.

        Parameters
        ----------
        plane :
            Input plane.
        lock :
            If given, block reading the plane into memory until the lock
            can be acquired.

        Returns
        -------
        plane :
            Thresholded plane.
        inside_brain_tiles :
            Boolean mask indicating which tiles are inside (1) or
            outside (0) the brain.
        """
        laplace_gaussian_sigma = self.log_sigma_size * self.soma_diameter
        planes = torch.as_tensor(
            np.moveaxis(planes.astype(np.float32), 2, 1),
            dtype=torch.float32,
            device="cuda",
        )
        torch.clip_(planes, 0, self.clipping_value)

        # Get tiles that are within the brain
        walker = TileWalker(planes, self.soma_diameter)
        walker.mark_bright_tiles()
        inside_brain_tiles = walker.bright_tiles_mask

        # Threshold the image
        thresholded_img = enhance_peaks(
            planes,
            self.clipping_value,
            gaussian_sigma=laplace_gaussian_sigma,
        )

        planes_1d = thresholded_img.view(thresholded_img.shape[0], -1)
        avg = torch.mean(planes_1d, dim=1, keepdim=True).unsqueeze(2)
        sd = torch.std(planes_1d, dim=1, keepdim=True).unsqueeze(2)
        threshold = avg + self.n_sds_above_mean_thresh * sd

        planes[thresholded_img > threshold] = self.threshold_value

        return planes, inside_brain_tiles

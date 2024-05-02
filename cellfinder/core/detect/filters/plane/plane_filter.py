from dataclasses import dataclass, field
from typing import Tuple

import torch

from cellfinder.core.detect.filters.plane.classical_filter import PeakEnchancer
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker
from cellfinder.core.detect.filters.setup_filters import DetectionSettings


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
    n_sds_above_mean_thresh: float

    peak_enhancer: PeakEnchancer = field(init=False)
    tile_walker: TileWalker = field(init=False)

    def __init__(self, settings: DetectionSettings):
        self.clipping_value = settings.clipping_value
        self.threshold_value = settings.threshold_value
        self.n_sds_above_mean_thresh = settings.n_sds_above_mean_thresh

        laplace_gaussian_sigma = (
            settings.log_sigma_size * settings.soma_diameter
        )
        self.peak_enhancer = PeakEnchancer(
            device=settings.torch_device,
            dtype=getattr(torch, settings.filterting_dtype),
            clipping_value=self.clipping_value,
            laplace_gaussian_sigma=laplace_gaussian_sigma,
        )

        self.tile_walker = TileWalker(
            plane_shape=settings.plane_shape,
            soma_diameter=settings.soma_diameter,
        )

    def get_tile_mask(
        self, planes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        torch.clip_(planes, 0, self.clipping_value)
        # Get tiles that are within the brain
        inside_brain_tiles = self.tile_walker.get_bright_tiles(planes)
        # Threshold the image
        enhanced_planes = self.peak_enhancer.enhance_peaks(planes)

        _threshold_planes(
            enhanced_planes, self.n_sds_above_mean_thresh, self.threshold_value
        )

        return planes, inside_brain_tiles


@torch.jit.script
def _threshold_planes(
    enhanced_planes: torch.Tensor,
    n_sds_above_mean_thresh: float,
    threshold_value: int,
) -> None:
    planes_1d = enhanced_planes.view(enhanced_planes.shape[0], -1)

    # add back last dim
    avg = torch.mean(planes_1d, dim=1, keepdim=True).unsqueeze(2)
    sd = torch.std(planes_1d, dim=1, keepdim=True).unsqueeze(2)
    threshold = avg + n_sds_above_mean_thresh * sd

    enhanced_planes[enhanced_planes > threshold] = threshold_value

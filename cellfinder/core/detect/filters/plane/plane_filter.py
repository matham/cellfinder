from typing import Tuple

import torch

from cellfinder.core.detect.filters.plane.classical_filter import PeakEnhancer
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker


class PlaneFilter:
    """
    Processor that filters each plane to highlight the peaks and also
    tiles and thresholds each plane returning a mask indicating which
    tiles are inside the brain.

    Each input plane is:

    1. Clipped to [0, clipping_value].
    2. Tiled and compared to the corner tile. Any tile that is "bright"
       according to `TileWalker` is marked as being in the brain.
    3. Filtered
       1. Run through the peak enhancement filter (see `PeakEnhancer`)
       2. Thresholded. Any values that are larger than
          (mean + stddev * n_sds_above_mean_thresh) are set to
          threshold_value.

    Parameters
    ----------
    clipping_value : int
        Upper value that the input planes are clipped to. Result is scaled so
        max is this value.
    threshold_value : int
        Value used to mark bright features in the input planes after they have
        been run through the 2D filter.
    n_sds_above_mean_thresh : float
        Number of standard deviations above the mean threshold to use for
        determining whether a voxel is bright.
    log_sigma_size : float
        Size of the Gaussian sigma for the Laplacian of Gaussian filtering.
    soma_diameter : float
        Diameter of the soma in voxels.
    torch_device: str
        The device on which the data and processing occurs on. Can be e.g.
        "cpu", "cuda" etc. Any data passed to the filter must be on this
        device. Returned data will also be on this device.
    dtype : str
        The data-type of the input planes and the type to use internally.
        E.g. "float32".
    use_scipy : bool
        If running on the CPU whether to use the scipy filters or the same
        pytorch filters used on CUDA. Scipy filters can be faster.
    """

    # Upper value that the input plane is clipped to
    clipping_value: int

    # filter that finds the peaks in the planes
    peak_enhancer: PeakEnhancer = None
    # generates tiles of the planes, with each tile marked as being inside
    # or outside the brain based on brightness
    tile_walker: TileWalker = None

    threshold_filter: "ThresholdFilter" = None

    def __init__(
        self,
        plane_shape: Tuple[int, int],
        clipping_value: int,
        threshold_value: int,
        n_sds_above_mean_thresh: float,
        log_sigma_size: float,
        soma_diameter: int,
        torch_device: str,
        dtype: str,
        use_scipy: bool,
    ):
        self.clipping_value = clipping_value

        self.threshold_filter = ThresholdFilter(
            threshold_value, n_sds_above_mean_thresh
        )

        laplace_gaussian_sigma = log_sigma_size * soma_diameter
        self.peak_enhancer = PeakEnhancer(
            torch_device=torch_device,
            dtype=getattr(torch, dtype),
            clipping_value=self.clipping_value,
            laplace_gaussian_sigma=laplace_gaussian_sigma,
            use_scipy=use_scipy,
        )

        self.tile_walker = TileWalker(
            plane_shape=plane_shape,
            soma_diameter=soma_diameter,
        )

    def clip_input(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Clips the input planes, in-place, to specified `clipping_value`.

        Parameters
        ----------
        planes : torch.Tensor
            Input planes (z-stack).

        Returns
        -------
        clipped planes : torch.Tensor
            Same tensor as the input `planes`.
        """
        torch.clip_(planes, 0, self.clipping_value)
        return planes

    def smooth_planes(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Takes previously clipped planes and smooths them using median ang
        Gaussian filtering.

        Parameters
        ----------
        planes : torch.Tensor
            Input planes (z-stack). It is not modified.

        Returns
        -------
        smoothed_planes : torch.Tensor
            Smoothed planes (z-stack).
        """
        return self.peak_enhancer.smooth_planes(planes)

    def peak_enhance_planes(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Takes previously clipped planes and enhances the peaks.

        Parameters
        ----------
        planes : torch.Tensor
            Input planes (z-stack). It is not modified.

        Returns
        -------
        peak_enhanced_planes : torch.Tensor
            Peak enhanced planes (z-stack).
        """
        return self.peak_enhancer.enhance_peaks(planes)

    def threshold_peak_enhanced_planes(
        self, peak_enhanced_planes: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes previously clipped and peak enhanced planes and thresholds them
        a plane at a time.

        Parameters
        ----------
        peak_enhanced_planes : torch.Tensor
            Peak enhanced planes (z-stack). It is not modified.

        Returns
        -------
        thresholded planes : torch.Tensor
            New tensor.
        """
        planes = self.threshold_filter.threshold_planes(
            peak_enhanced_planes,
        )
        return planes

    def get_inside_mask(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Takes previously clipped planes and estimates the tiles that are
        inside/outside the brain.

        Parameters
        ----------
        planes : torch.Tensor
            Input planes (z-stack). It is not modified.

        Returns
        -------
        peak_enhanced_planes : torch.Tensor
            Peak enhanced planes (z-stack).
        """
        return self.tile_walker.get_bright_tiles(planes)

    def get_tiled_buffer(self, depth: int, device: str):
        return self.tile_walker.get_tiled_buffer(depth, device)


class ThresholdFilter:

    # Value used to mark bright features in the input planes after they have
    # been run through the 2D filter
    threshold_value: int
    # voxels who are this many std above mean or more are set to
    # threshold_value
    n_sds_above_mean_thresh: float

    def __init__(
        self,
        threshold_value: int,
        n_sds_above_mean_thresh: float,
    ):
        self.threshold_value = threshold_value
        self.n_sds_above_mean_thresh = n_sds_above_mean_thresh

    def threshold_planes(self, enhanced_planes: torch.Tensor) -> torch.Tensor:
        planes = _threshold_planes(
            enhanced_planes,
            self.n_sds_above_mean_thresh,
            self.threshold_value,
        )
        return planes


@torch.jit.script
def _threshold_planes(
    enhanced_planes: torch.Tensor,
    n_sds_above_mean_thresh: float,
    threshold_value: int,
) -> torch.Tensor:
    """
    Sets each pixel in the returned planes to threshold_value, where the
    corresponding enhanced_plane > mean + n_sds_above_mean_thresh*std.
    Each pixel will be set to zero elsewhere. Original `planes` is unchanged.
    """
    z, y, x = enhanced_planes.shape

    # ---- get per-plane global threshold ----
    planes_1d = enhanced_planes.view(z, -1)
    # add back last dim
    std, mean = torch.std_mean(planes_1d, dim=1, keepdim=True)
    threshold = mean.unsqueeze(2) + n_sds_above_mean_thresh * std.unsqueeze(2)

    # subsequent steps only care about the values that are set to threshold or
    # above in planes. We set values in *planes* to threshold based on the
    # value in *enhanced_planes*. So, there could be values in planes that are
    # at threshold already, but in enhanced_planes they are not. So it's best
    # to zero all other values, so voxels previously at threshold don't count
    planes = torch.zeros_like(enhanced_planes)
    planes[enhanced_planes > threshold] = threshold_value
    return planes

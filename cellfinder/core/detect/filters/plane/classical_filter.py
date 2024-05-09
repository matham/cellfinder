import torch
import torch.nn.functional as F
from kornia.filters.kernels import (
    get_binary_kernel2d,
    get_gaussian_kernel1d,
    get_laplacian_kernel2d,
    normalize_kernel2d,
)


@torch.jit.script
def normalize(filtered_planes: torch.Tensor, clipping_value: float) -> None:
    """
    Normalizes the 3d tensor so each z-plane is independantly scaled to be
    in the [0, clipping_value] range.

    It it done to filtered_planes inplace.
    """
    num_z = filtered_planes.shape[0]
    filtered_planes_1d = filtered_planes.view(num_z, -1)

    filtered_planes_1d.mul_(-1)

    planes_min = torch.min(filtered_planes_1d, dim=1, keepdim=True)[0]
    filtered_planes_1d.sub_(planes_min)
    # take max after subtraction
    planes_max = torch.max(filtered_planes_1d, dim=1, keepdim=True)[0]
    filtered_planes_1d.div_(planes_max)

    # To leave room to label in the 3d detection.
    filtered_planes_1d.mul_(clipping_value)


@torch.jit.script
def filter_for_peaks(
    planes: torch.Tensor,
    bin_kernel: torch.Tensor,
    gauss_kernel: torch.Tensor,
    lap_kernel: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Takes the 3d z-stack and returns a new z-stack where the peaks are
    highlighted.

    It applies a median filter -> gaussian filter -> laplacian filter.
    """
    filtered_planes = planes.unsqueeze(1)  # ZYX -> ZCYX input

    # ---------- median filter ----------
    # extract patches to compute median over for each pizel
    # We go from ZCYX -> ZCYX, C=1 to C=9 and containing the elements around
    # each Z,X,Y over which we compute the median
    filtered_planes = F.conv2d(filtered_planes, bin_kernel, padding="same")
    # we're going back to ZCYX=Z1YX
    filtered_planes = filtered_planes.median(dim=1, keepdim=True)[0]

    # ---------- gaussian filter ----------
    # We apply the 1D gaussian filter twice, once for Y and once for X. The
    # filter shape passed in is 11K1 or 111K, depending on device. Where
    # K=filter size
    # todo use reflect padding or check if needed
    # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-with-1d-
    # filter-along-a-dim/201734/2 for the reason for the moveaxis depending
    # on the device
    if device == "cpu":
        # kernel shape is 11K1. First do Y (second to last axis)
        filtered_planes = F.conv2d(
            filtered_planes, gauss_kernel, padding="same"
        )
        # To do X, exchange X,Y axis, filter, change back. On CPU, Y (second
        # to last) axis is faster.
        filtered_planes = F.conv2d(
            filtered_planes.moveaxis(-1, -2), gauss_kernel, padding="same"
        ).moveaxis(-1, -2)
    else:
        # kernel shape is 111K
        # First do Y (second to last axis). Exchange X,Y axis, filter, change
        # back. On CUDA, X (last) axis is faster.
        filtered_planes = F.conv2d(
            filtered_planes.moveaxis(-1, -2), gauss_kernel, padding="same"
        ).moveaxis(-1, -2)
        # now do X, last axis
        filtered_planes = F.conv2d(
            filtered_planes, gauss_kernel, padding="same"
        )

    # ---------- laplacian filter ----------
    # filter comes in as 2d, make it 4d as required for conv
    lap_kernel = lap_kernel.view(
        1, 1, lap_kernel.shape[0], lap_kernel.shape[1]
    )
    # todo use reflect padding or check if needed
    filtered_planes = F.conv2d(filtered_planes, lap_kernel, padding="same")

    # we don't need the channel axis
    return filtered_planes[:, 0, :, :]


class PeakEnchancer:
    """
    A class that filters each plane in a z-stack such that peaks are
    visualized.

    It uses a series of 2D filters of median -> gaussian ->
    laplacian. Then normalizes each plane to be between [0, clipping_value].

    Parameters
    ----------
    torch_device: str
        The device on which the data and processing occurs on. Can be e.g.
        "cpu", "cuda" etc. Any data passed to the filter must be on this
        device. Returned data will also be on this device.
    dtype : torch.dtype
        The data-type of the input planes and the type to use internally.
        E.g. `torch.float32`.
    clipping_value : int
        The value such that after normalizing, the max value will be this
        clipping_value.
    laplace_gaussian_sigma : float
        Size of the sigma for the gaussian filter.
    """

    # binary filter that gets square patches for each voxel so we can find the
    # median. Shape is ZCYX (C=1). See filter_for_peaks for details.
    bin_kernel: torch.Tensor

    # gaussian 1D kernel of shape 11K1 or 111K, depending on device. Where
    # K=filter size. See filter_for_peaks for details.
    gauss_kernel: torch.Tensor

    # 2D laplacian kernel of shape KxK. Where
    # K=filter size. See filter_for_peaks for details.
    lap_kernel: torch.Tensor

    # the value such that after normalizing, the max value will be this
    # clipping_value
    clipping_value: float

    # the torch device to run on. E.g. cpu/cuda.
    torch_device: str

    def __init__(
        self,
        torch_device: str,
        dtype: torch.dtype,
        clipping_value: float,
        laplace_gaussian_sigma: float,
    ):
        super().__init__()
        self.torch_device = torch_device.lower()
        self.clipping_value = clipping_value

        # must be odd kernel
        self.bin_kernel = get_binary_kernel2d(
            3, device=torch_device, dtype=dtype
        )

        kernel_size = 2 * int(round(4 * laplace_gaussian_sigma)) + 1
        # shape is 11K1
        self.gauss_kernel = get_gaussian_kernel1d(
            kernel_size,
            laplace_gaussian_sigma,
            device=torch_device,
            dtype=dtype,
        ).view(1, 1, -1, 1)
        # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-
        # with-1d-filter-along-a-dim/201734. Conv2d is faster on a specific dim
        # for 1D filters depending on CPU/CUDA. See also filter_for_peaks
        if self.torch_device == "cpu":
            # on CPU, we only do conv2d on the (third) dim
            self.gauss_kernel = self.gauss_kernel.view(1, 1, -1, 1)
        else:
            # on CUDA, we only do conv2d on the (forth) dim
            self.gauss_kernel = self.gauss_kernel.view(1, 1, 1, -1)

        lap_kernel = get_laplacian_kernel2d(
            3, device=torch_device, dtype=dtype
        )
        # it is 2d
        self.lap_kernel = normalize_kernel2d(lap_kernel)

    def enhance_peaks(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Applies the filtering and normalization to the 3d z-stack (not inplace)
        and returns the filtered z-stack.
        """
        filtered_planes = filter_for_peaks(
            planes,
            self.bin_kernel,
            self.gauss_kernel,
            self.lap_kernel,
            self.torch_device,
        )
        normalize(filtered_planes, self.clipping_value)
        return filtered_planes

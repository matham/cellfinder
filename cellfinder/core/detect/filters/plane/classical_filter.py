import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import (
    get_binary_kernel2d,
    get_gaussian_kernel1d,
    get_laplacian_kernel2d,
    normalize_kernel2d,
)
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d


@torch.jit.script
def normalize(filtered_planes: torch.Tensor, clipping_value: float) -> None:
    """
    Normalizes the 3d tensor so each z-plane is independently scaled to be
    in the [0, clipping_value] range.

    It is done to filtered_planes inplace.
    """
    num_z = filtered_planes.shape[0]
    filtered_planes_1d = filtered_planes.view(num_z, -1)

    filtered_planes_1d.mul_(-1)

    planes_min = torch.min(filtered_planes_1d, dim=1, keepdim=True)[0]
    filtered_planes_1d.sub_(planes_min)
    # take max after subtraction
    planes_max = torch.max(filtered_planes_1d, dim=1, keepdim=True)[0]
    # if min = max = zero, divide by 1 - it'll stay zero
    planes_max[planes_max == 0] = 1
    filtered_planes_1d.div_(planes_max)

    # To leave room to label in the 3d detection.
    filtered_planes_1d.mul_(clipping_value)


def filter_for_peaks(
    planes: torch.Tensor,
    med_filter: nn.Module,
    gauss_filter: nn.Module,
    lap_filter: nn.Module,
    device: str,
) -> torch.Tensor:
    """
    Takes the 3d z-stack and returns a new z-stack where the peaks are
    highlighted.

    It applies a median filter -> gaussian filter -> laplacian filter.
    """
    filtered_planes = planes.unsqueeze(1)  # ZYX -> ZCYX input

    # ---------- median filter ----------
    # extracts patches to compute median over for each pixel
    # We go from ZCYX -> ZCYX, C=1 to C=9 and containing the elements around
    # each Z,X,Y over which we compute the median
    filtered_planes = med_filter(filtered_planes)
    # we're going back to ZCYX=Z1YX by taking median of patches
    filtered_planes = filtered_planes.median(dim=1, keepdim=True)[0]

    # ---------- gaussian filter ----------
    # We apply the 1D gaussian filter twice, once for Y and once for X. The
    # filter shape passed in is 11K1 or 111K, depending on device. Where
    # K=filter size
    # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-with-1d-
    # filter-along-a-dim/201734/2 for the reason for the moveaxis depending
    # on the device
    if device == "cpu":
        # kernel shape is 11K1. First do Y (second to last axis)
        filtered_planes = gauss_filter(filtered_planes)
        # To do X, exchange X,Y axis, filter, change back. On CPU, Y (second
        # to last) axis is faster.
        filtered_planes = gauss_filter(
            filtered_planes.moveaxis(-1, -2)
        ).moveaxis(-1, -2)
    else:
        # kernel shape is 111K
        # First do Y (second to last axis). Exchange X,Y axis, filter, change
        # back. On CUDA, X (last) axis is faster.
        filtered_planes = gauss_filter(
            filtered_planes.moveaxis(-1, -2)
        ).moveaxis(-1, -2)
        # now do X, last axis
        filtered_planes = gauss_filter(filtered_planes)

    # ---------- laplacian filter ----------
    # it's a 2d filter
    filtered_planes = lap_filter(filtered_planes)

    # we don't need the channel axis
    return filtered_planes[:, 0, :, :]


class PeakEnhancer:
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
    use_scipy : bool
        If running on the CPU whether to use the scipy filters or the same
        pytorch filters used on CUDA. Scipy filters can be faster.
    """

    # binary filter that generates square patches for each voxel so we can find
    # the median. Input shape expected is ZCYX (C=1)
    med_filter: nn.Module

    # gaussian 1D filter with kernel/weight shape 11K1 or 111K, depending
    # on device. Where K=filter size
    gauss_filter: nn.Module

    # 2D laplacian filter with kernel/weight shape KxK. Where
    # K=filter size
    lap_filter: nn.Module

    # the value such that after normalizing, the max value will be this
    # clipping_value
    clipping_value: float

    # sigma value for gaussian filter
    laplace_gaussian_sigma: float

    # the torch device to run on. E.g. cpu/cuda.
    torch_device: str

    # when running on CPU whether to use pytorch or scipy for filters
    use_scipy: bool

    median_filter_size: int = 3
    """
    The median filter size in x/y direction.

    **Must** be odd.
    """

    laplace_filter_size: int = 3
    """
    The laplacian filter size in x/y direction.

    **Must** be odd.
    """

    _padding: int
    """
    The padding to apply to the input data so the output will be the same as
    the input after filtering.

    During filtering we then just use the "valid" area of the filtered data.
    """

    def __init__(
        self,
        torch_device: str,
        dtype: torch.dtype,
        clipping_value: float,
        laplace_gaussian_sigma: float,
        use_scipy: bool,
    ):
        super().__init__()
        self.torch_device = torch_device.lower()
        self.clipping_value = clipping_value
        self.laplace_gaussian_sigma = laplace_gaussian_sigma
        self.use_scipy = use_scipy

        if not (self.median_filter_size % 2):
            raise ValueError("The median filter size must be odd")
        if not (self.laplace_filter_size % 2):
            raise ValueError("The laplacian filter size must be odd")
        assert self.gaussian_filter_size % 2, "Should be odd"

        self._padding = (
            self.median_filter_size
            - 1
            + self.gaussian_filter_size
            - 1
            + self.laplace_filter_size
            - 1
        ) // 2

        self.med_filter = self._get_median_filter(torch_device, dtype)
        self.gauss_filter = self._get_gaussian_filter(
            torch_device, dtype, laplace_gaussian_sigma
        )
        self.lap_filter = self._get_laplacian_filter(torch_device, dtype)

    @property
    def gaussian_filter_size(self) -> int:
        """
        The gaussian filter 1d size.

        It is odd.
        """
        return 2 * int(round(4 * self.laplace_gaussian_sigma)) + 1

    def _get_median_filter(
        self, torch_device: str, dtype: torch.dtype
    ) -> nn.Module:
        """
        Gets a median patch generator torch module, already on the correct
        device.
        """
        # must be odd kernel
        kernel_n = self.median_filter_size
        weight = get_binary_kernel2d(
            kernel_n, device=torch_device, dtype=dtype
        )

        # extract patches to compute median over for each pixel. When passing
        # input we go from ZCYX -> ZCYX, C=1 to C=9 and containing the elements
        # around each Z,X,Y over which we can then compute the median
        module = nn.Conv2d(
            1,
            kernel_n * kernel_n,
            (kernel_n, kernel_n),
            padding="valid",
            bias=False,
            device=torch_device,
            dtype=dtype,
        )
        module.weight.copy_(weight)

        return module

    def _get_gaussian_filter(
        self,
        torch_device: str,
        dtype: torch.dtype,
        laplace_gaussian_sigma: float,
    ) -> nn.Module:
        kernel_size = self.gaussian_filter_size
        # shape of kernel is 11K1 with dims Z, C, Y, X. C=1, Z is expanded to
        # number of z.
        gauss_kernel = get_gaussian_kernel1d(
            kernel_size,
            laplace_gaussian_sigma,
            device=torch_device,
            dtype=dtype,
        ).view(1, 1, -1, 1)
        # default shape is y, x with y axis filtered only - we flip input to
        # filter on x
        kernel_shape = kernel_size, 1

        # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-
        # with-1d-filter-along-a-dim/201734. Conv2d is faster on a specific dim
        # for 1D filters depending on CPU/CUDA. See also filter_for_peaks
        # on CPU, we only do conv2d on the (1st) dim
        if torch_device != "cpu":
            # on CUDA, we only filter on the x dim, flipping input to filter y
            gauss_kernel = gauss_kernel.view(1, 1, 1, -1)
            kernel_shape = 1, kernel_size

        module = nn.Conv2d(
            1,
            1,
            kernel_shape,
            padding="valid",
            bias=False,
            device=torch_device,
            dtype=dtype,
        )
        module.weight.copy_(gauss_kernel)

        return module

    def _get_laplacian_filter(
        self, torch_device: str, dtype: torch.dtype
    ) -> nn.Module:
        # must be odd kernel
        lap_kernel = get_laplacian_kernel2d(
            self.laplace_filter_size, device=torch_device, dtype=dtype
        )
        # it is 2d so turn kernel in 4d, like other filters (ZCYX)
        lap_kernel = normalize_kernel2d(lap_kernel).unsqueeze(0).unsqueeze(0)

        module = nn.Conv2d(
            1,
            1,
            (self.laplace_filter_size, self.laplace_filter_size),
            padding="valid",
            bias=False,
            device=torch_device,
            dtype=dtype,
        )
        module.weight.copy_(lap_kernel)

        return module

    def enhance_peaks(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Applies the filtering and normalization to the 3d z-stack (not inplace)
        and returns the filtered z-stack.
        """
        if self.torch_device == "cpu" and self.use_scipy:
            filtered_planes = planes.clone()
            for i in range(planes.shape[0]):
                img = planes[i, :, :].numpy()
                img = medfilt2d(img)
                img = gaussian_filter(img, self.laplace_gaussian_sigma)
                img = laplace(img)
                filtered_planes[i, :, :] = torch.from_numpy(img)
        else:
            # pad enough for all the filters
            padding = self._padding
            mode = "reflect"
            # if the input is too small for reflection padding, use replication
            if planes.shape[-2] < padding or planes.shape[-1] < padding:
                mode = "replicate"
            # left, right, top, bottom padding
            filtered_planes = F.pad(planes, (padding,) * 4, mode)

            filtered_planes = filter_for_peaks(
                filtered_planes,
                self.med_filter,
                self.gauss_filter,
                self.lap_filter,
                self.torch_device,
            )

        normalize(filtered_planes, self.clipping_value)
        return filtered_planes

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
    filtered_planes = planes.unsqueeze(1)  # ZXY -> ZCXY input
    # extract patches to compute median over for each pizel
    filtered_planes = F.conv2d(filtered_planes, bin_kernel, padding="same")
    # ZCXY -> ZCXY. C channel contains the elements to compute median over
    # so that dimension becomes size 1 again
    filtered_planes = filtered_planes.median(dim=1, keepdim=True)[0]

    # 2d kernel to 1d. Shape is 11K1 or 111K, depending on device
    # todo use reflect padding or check if needed
    if device == "cpu":
        # kernel shape is 11K1, so to conv Y, move it to third dim
        filtered_planes = F.conv2d(
            filtered_planes, gauss_kernel, padding="same"
        )
        filtered_planes = F.conv2d(
            filtered_planes.moveaxis(-1, -2), gauss_kernel, padding="same"
        ).moveaxis(-1, -2)
    else:
        # kernel shape is 111K so to conv X, move it to fourth dim
        filtered_planes = F.conv2d(
            filtered_planes.moveaxis(-1, -2), gauss_kernel, padding="same"
        ).moveaxis(-1, -2)
        filtered_planes = F.conv2d(
            filtered_planes, gauss_kernel, padding="same"
        )

    # it is 2d
    lap_kernel = lap_kernel.view(
        1, 1, lap_kernel.shape[0], lap_kernel.shape[1]
    )
    # todo use reflect padding or check if needed
    filtered_planes = F.conv2d(filtered_planes, lap_kernel, padding="same")

    return filtered_planes[:, 0, :, :]


class PeakEnchancer:

    # Shape is ZCXY of binary to get square patches for median
    bin_kernel: torch.Tensor

    # K=kernel size. Then for 1d we get a 2D kernel of 1K
    gauss_kernel: torch.Tensor

    lap_kernel: torch.Tensor

    clipping_value: float

    device: str

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        clipping_value: float,
        laplace_gaussian_sigma: float,
    ):
        super().__init__()
        self.device = device.lower()
        self.clipping_value = clipping_value

        # must be odd kernel
        self.bin_kernel = get_binary_kernel2d(3, device=device, dtype=dtype)

        kernel_size = 2 * int(round(4 * laplace_gaussian_sigma)) + 1
        # shape is 11K1
        self.gauss_kernel = get_gaussian_kernel1d(
            kernel_size, laplace_gaussian_sigma, device=device, dtype=dtype
        ).view(1, 1, -1, 1)
        # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-
        # with-1d-filter-along-a-dim/201734. Conv2d is faster on a specific dim
        # for 1D filters depending on CPU/CUDA
        if self.device == "cpu":
            # on CPU, we only do conv2d on the x (third) dim
            self.gauss_kernel = self.gauss_kernel.view(1, 1, -1, 1)
        else:
            # on CUDA, we only do conv2d on the y (forth) dim
            self.gauss_kernel = self.gauss_kernel.view(1, 1, 1, -1)

        lap_kernel = get_laplacian_kernel2d(3, device=device, dtype=dtype)
        # it is 2d
        self.lap_kernel = normalize_kernel2d(lap_kernel)

    def enhance_peaks(self, planes: torch.Tensor) -> torch.Tensor:
        filtered_planes = filter_for_peaks(
            planes,
            self.bin_kernel,
            self.gauss_kernel,
            self.lap_kernel,
            self.device,
        )
        normalize(filtered_planes, self.clipping_value)
        return filtered_planes

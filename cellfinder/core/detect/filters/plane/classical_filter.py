import torch
from kornia.filters.gaussian import gaussian_blur2d
from kornia.filters.laplacian import laplacian
from kornia.filters.median import median_blur


def enhance_peaks(
    planes: torch.Tensor, clipping_value: float, gaussian_sigma: float = 2.5
) -> torch.Tensor:
    """
    Enhances the peaks (bright pixels) in an input image.

    Parameters:
    ----------
    img : np.ndarray
        Input image.
    clipping_value : float
        Maximum value for the enhanced image.
    gaussian_sigma : float, optional
        Standard deviation for the Gaussian filter. Default is 2.5.

    Returns:
    -------
    np.ndarray
        Enhanced image with peaks.

    Notes:
    ------
    The enhancement process includes the following steps:
    1. Applying a 2D median filter.
    2. Applying a Laplacian of Gaussian filter (LoG).
    3. Multiplying by -1 (bright spots respond negative in a LoG).
    4. Rescaling image values to range from 0 to clipping value.
    """
    num_z = planes.shape[0]

    filtered_planes = planes.unsqueeze(1)
    filtered_planes = median_blur(filtered_planes, kernel_size=3)
    filtered_planes = gaussian_blur2d(
        filtered_planes,
        2 * int(round(4 * gaussian_sigma)) + 1,
        (gaussian_sigma, gaussian_sigma),
    )
    filtered_planes = laplacian(filtered_planes, kernel_size=3)

    filtered_planes = filtered_planes[:, 0, :, :]
    filtered_planes.mul_(-1)

    filtered_planes_1d = filtered_planes.view(num_z, -1)

    planes_min = torch.min(filtered_planes_1d, dim=1, keepdim=True)[
        0
    ].unsqueeze(2)
    filtered_planes.sub_(planes_min)

    # take max after subtraction
    planes_max = torch.max(filtered_planes_1d, dim=1, keepdim=True)[
        0
    ].unsqueeze(2)
    filtered_planes.div_(planes_max)

    # To leave room to label in the 3d detection.
    filtered_planes.mul_(clipping_value)
    return filtered_planes

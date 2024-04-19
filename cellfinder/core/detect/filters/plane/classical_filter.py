import torch
from kornia.filters.gaussian import gaussian_blur2d
from kornia.filters.laplacian import laplacian
from kornia.filters.median import median_blur


def enhance_peaks(
    img: torch.Tensor, clipping_value: float, gaussian_sigma: float = 2.5
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
    img = img.unsqueeze(0).unsqueeze(0)
    filtered_img = median_blur(img, kernel_size=3)
    filtered_img = gaussian_blur2d(
        filtered_img,
        2 * int(round(4 * gaussian_sigma)) + 1,
        (gaussian_sigma, gaussian_sigma),
    )
    filtered_img = laplacian(filtered_img, kernel_size=3)

    filtered_img.mul_(-1)
    filtered_img.sub_(filtered_img.min())
    filtered_img.div_(filtered_img.max())
    # To leave room to label in the 3d detection.
    filtered_img.mul_(clipping_value)
    return filtered_img[0, 0, :, :]

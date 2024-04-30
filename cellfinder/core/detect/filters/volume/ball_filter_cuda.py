import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.tools.array_operations import bin_mean_3d
from cellfinder.core.tools.geometry import make_sphere

DEBUG = False


@lru_cache(maxsize=50)
def get_kernel(ball_xy_size: int, ball_z_size: int) -> np.ndarray:
    # Create a spherical kernel.
    #
    # This is done by:
    # 1. Generating a binary sphere at a resolution *upscale_factor* larger
    #    than desired.
    # 2. Downscaling the binary sphere to get a 'fuzzy' sphere at the
    #    original intended scale
    upscale_factor: int = 7
    upscaled_kernel_shape = (
        upscale_factor * ball_xy_size,
        upscale_factor * ball_xy_size,
        upscale_factor * ball_z_size,
    )
    upscaled_ball_centre_position = (
        np.floor(upscaled_kernel_shape[0] / 2),
        np.floor(upscaled_kernel_shape[1] / 2),
        np.floor(upscaled_kernel_shape[2] / 2),
    )
    upscaled_ball_radius = upscaled_kernel_shape[0] / 2.0

    sphere_kernel = make_sphere(
        upscaled_kernel_shape,
        upscaled_ball_radius,
        upscaled_ball_centre_position,
    )
    sphere_kernel = sphere_kernel.astype(np.float32)
    kernel = bin_mean_3d(
        sphere_kernel,
        bin_width=upscale_factor,
        bin_height=upscale_factor,
        bin_depth=upscale_factor,
    )

    assert (
        kernel.shape[2] == ball_z_size
    ), "Kernel z dimension should be {}, got {}".format(
        ball_z_size, kernel.shape[2]
    )

    return kernel


class BallFilter:
    """
    A 3D ball filter.

    This runs a spherical kernel across the (x, y) dimensions
    of a *ball_z_size* stack of planes, and marks pixels in the middle
    plane of the stack that have a high enough intensity within the
    spherical kernel.
    """

    num_batches_before_ready: int

    def __init__(self, settings: DetectionSettings):
        """
        Parameters
        ----------
        """
        self.settings = settings

        self.ball_xy_size = settings.ball_xy_size
        self.ball_z_size = settings.ball_z_size
        self.overlap_fraction = settings.ball_overlap_fraction
        self.tile_step_dim1 = settings.tile_dim1
        self.tile_step_dim2 = settings.tile_dim2

        self.THRESHOLD_VALUE = settings.threshold_value
        self.SOMA_CENTRE_VALUE = settings.soma_centre_value

        kernel = np.moveaxis(
            get_kernel(self.ball_xy_size, self.ball_z_size), 2, 0
        )
        self.overlap_threshold = np.sum(self.overlap_fraction * kernel)
        self.kernel_xy_size = kernel.shape[-2:]
        self.kernel_z_size = self.ball_z_size

        kernel = (
            torch.from_numpy(kernel).type(settings.torch_dtype).pin_memory()
        )
        self.kernel = (
            kernel.unsqueeze(0)
            .unsqueeze(0)
            .to(device=settings.torch_device, non_blocking=True)
        )

        # Stores the current planes that are being filtered
        # first axis is z for faster rotating the z-axis
        if settings.batch_size < self.ball_z_size:
            raise ValueError(
                f"batch_size={settings.batch_size} < "
                f"ball_z_size (kernel)={self.ball_z_size}"
            )
        self.num_batches_before_ready = 1

        self.volume = torch.empty(
            (0, settings.plane_dim1, settings.plane_dim2),
            dtype=settings.torch_dtype,
        )
        # Index of the middle plane in the volume
        self.middle_z_idx = int(np.floor(self.ball_z_size / 2))

        # first axis is z
        tile_dim1 = int(np.ceil(settings.plane_dim1 / self.tile_step_dim1))
        tile_dim2 = int(np.ceil(settings.plane_dim2 / self.tile_step_dim2))
        self.inside_brain_tiles = torch.empty(
            (
                0,
                tile_dim1,
                tile_dim2,
            ),
            dtype=torch.bool,
        )

        # Get extents of image that are covered by tiles
        tile_mask_covered_img_dim1 = tile_dim1 * self.tile_step_dim1
        tile_mask_covered_img_dim2 = tile_dim2 * self.tile_step_dim2
        # Get maximum offsets for the ball within the tiled plane
        max_dim1 = tile_mask_covered_img_dim1 - self.ball_xy_size
        max_dim2 = tile_mask_covered_img_dim2 - self.ball_xy_size
        self.tiled_xy = max_dim1, max_dim2

    @property
    def first_valid_plane(self):
        # todo: double check this is accurate
        return int(math.floor(self.ball_z_size / 2))

    @property
    def ready(self) -> bool:
        """
        Return `True` if enough planes have been appended to run the filter.
        """
        return self.volume.shape[0] >= self.kernel_z_size

    def append(self, planes: np.ndarray, masks: np.ndarray) -> None:
        """
        Add a new 2D plane to the filter.
        """
        if self.volume.shape[0]:
            num_remaining = self.kernel_z_size - (self.middle_z_idx + 1)
            num_remaining_with_padding = num_remaining + self.middle_z_idx
            self.volume = torch.cat(
                [self.volume[-num_remaining_with_padding:, :, :], planes],
                dim=0,
            )
            self.inside_brain_tiles = torch.cat(
                [
                    self.inside_brain_tiles[
                        -num_remaining_with_padding:, :, :
                    ],
                    masks,
                ],
                dim=0,
            )
        else:
            self.volume = planes.clone()
            self.inside_brain_tiles = masks.clone()

    def get_middle_planes(self) -> np.ndarray:
        """
        Get the plane in the middle of self.volume.
        """
        num_processed = self.volume.shape[0] - self.kernel_z_size + 1
        assert num_processed
        middle = self.middle_z_idx
        planes = (
            self.volume[middle : middle + num_processed, :, :]
            .cpu()
            .numpy()
            .copy()
        )
        return planes

    def walk(self, parallel: bool = False) -> None:
        # **don't** pass parallel as keyword arg - numba struggles with it
        # Highly optimised because most time critical
        max_dim1, max_dim2 = self.tiled_xy
        num_process = self.volume.shape[0] - self.kernel_z_size + 1
        middle = self.middle_z_idx

        # inside = self.inside_brain_tiles[middle: middle + num_process, :, :]
        # _, w, h = inside.shape
        # inside = inside.view(w, 1, h).expand(w,
        # self.tile_step_dim1, h).contiguous().view(-1, h)
        # inside = self.inside_brain_tiles[middle : middle + num_process, :, :]
        # orig_w, orig_h = inside.shape[1:]
        # final_w = self.tile_step_dim1 * orig_w
        # inside = (
        #     inside
        #     .view(num_process, orig_w, 1, orig_h)
        #     .expand(num_process, orig_w, self.tile_step_dim1, orig_h)
        #     .contiguous()
        #     .view(num_process, -1, orig_h)
        #     .view(num_process, final_w, orig_h, 1)
        #     .expand(num_process, final_w, orig_h, self.tile_step_dim2)
        #     .contiguous()
        #     .view(num_process, final_w, -1)
        # )
        inside = (
            self.inside_brain_tiles[middle : middle + num_process, :, :]
            .repeat_interleave(self.tile_step_dim1, dim=1)
            .repeat_interleave(self.tile_step_dim2, dim=2)
        )
        # max_xxx may be larger than actual volume
        sub_volume = self.volume[:, :max_dim1, :max_dim2]
        # it may be larger if the last tile sticks outside the volume

        volume_tresh = (
            (sub_volume >= self.THRESHOLD_VALUE)
            .unsqueeze(0)
            .unsqueeze(0)
            .type(self.kernel.dtype)
        )
        # pherical kernel is symetric so convolution=corrolation
        overlaps = (
            F.conv3d(volume_tresh, self.kernel, stride=1)[0, 0, :, :, :]
            > self.overlap_threshold
        )
        inside = inside[:, : overlaps.shape[1], : overlaps.shape[2]]

        sub_volume[
            middle : middle + num_process,
            : overlaps.shape[1],
            : overlaps.shape[2],
        ][torch.logical_and(overlaps, inside)] = self.SOMA_CENTRE_VALUE

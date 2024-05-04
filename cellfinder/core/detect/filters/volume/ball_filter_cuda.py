import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.tools.array_operations import bin_mean_3d
from cellfinder.core.tools.geometry import make_sphere

DEBUG = False


class InvalidVolume(ValueError):
    pass


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

    inside_brain_tiles: Optional[torch.Tensor] = None

    def __init__(self, settings: DetectionSettings, use_mask: bool = True):
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

        d1 = settings.plane_dim1
        d2 = settings.plane_dim2
        ball_xy_size = self.ball_xy_size
        if d1 < ball_xy_size or d2 < ball_xy_size:
            raise InvalidVolume(
                f"Invalid plane size {d1}x{d2}. Needs to be at least "
                f"{ball_xy_size} in each dimension"
            )

        self.THRESHOLD_VALUE = settings.threshold_value
        self.SOMA_CENTRE_VALUE = settings.soma_centre_value

        kernel = np.moveaxis(get_kernel(ball_xy_size, self.ball_z_size), 2, 0)
        self.overlap_threshold = np.sum(self.overlap_fraction * kernel)
        self.kernel_xy_size = kernel.shape[-2:]
        self.kernel_z_size = self.ball_z_size

        kernel = (
            torch.from_numpy(kernel)
            .type(getattr(torch, settings.filterting_dtype))
            .pin_memory()
        )
        self.kernel = (
            kernel.unsqueeze(0)
            .unsqueeze(0)
            .to(device=settings.torch_device, non_blocking=True)
        )

        self.num_batches_before_ready = int(
            math.ceil(self.ball_z_size / settings.batch_size)
        )
        # Stores the current planes that are being filtered
        # first axis is z for faster rotating the z-axis
        self.volume = torch.empty(
            (0, settings.plane_dim1, settings.plane_dim2),
            dtype=getattr(torch, settings.filterting_dtype),
        )
        # Index of the middle plane in the volume
        self.middle_z_idx = int(np.floor(self.ball_z_size / 2))

        if not use_mask:
            return
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

    def append(
        self, planes: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> None:
        """
        Add a new 2D plane to the filter.
        """
        if self.volume.shape[0]:
            if self.volume.shape[0] < self.kernel_z_size:
                num_remaining_with_padding = 0
            else:
                num_remaining = self.kernel_z_size - (self.middle_z_idx + 1)
                num_remaining_with_padding = num_remaining + self.middle_z_idx

            self.volume = torch.cat(
                [self.volume[-num_remaining_with_padding:, :, :], planes],
                dim=0,
            )

            if self.inside_brain_tiles is not None:
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
            if self.inside_brain_tiles is not None:
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

    def walk(self) -> None:
        num_process = self.volume.shape[0] - self.kernel_z_size + 1
        dim1, dim2 = self.volume.shape[1:]
        middle = self.middle_z_idx

        volume_tresh = (
            (self.volume >= self.THRESHOLD_VALUE)
            .unsqueeze(0)
            .unsqueeze(0)
            .type(self.kernel.dtype)
        )
        # spherical kernel is symetric so convolution=corrolation
        overlaps = (
            F.conv3d(volume_tresh, self.kernel, stride=1)[0, 0, :, :, :]
            > self.overlap_threshold
        )

        # get only z's that are processed (e.g. with kernel=3, depth=5,
        # only 3 planes are processed). Also, only get the volume that is valid
        # - conv excludes edges
        dim1_valid, dim2_valid = overlaps.shape[1:]
        dim1_offset = (dim1 - dim1_valid) // 2
        dim2_offset = (dim2 - dim2_valid) // 2
        sub_volume = self.volume[
            middle : middle + num_process,
            dim1_offset : dim1_offset + dim1_valid,
            dim2_offset : dim2_offset + dim2_valid,
        ]

        if self.inside_brain_tiles is not None:
            # unfold tiles to cover the full area each tile covers
            inside = (
                self.inside_brain_tiles[middle : middle + num_process, :, :]
                .repeat_interleave(self.tile_step_dim1, dim=1)
                .repeat_interleave(self.tile_step_dim2, dim=2)
            )
            inside = inside[
                :,
                dim1_offset : dim1_offset + dim1_valid,
                dim2_offset : dim2_offset + dim2_valid,
            ]

            sub_volume[torch.logical_and(overlaps, inside)] = (
                self.SOMA_CENTRE_VALUE
            )

        else:
            sub_volume[overlaps] = self.SOMA_CENTRE_VALUE

import random

import numpy as np
import pytest
import torch

from cellfinder.core.tools import image_processing as img_tools


def test_crop_center_2d():
    x_shape = random.randint(2, 100)
    y_shape = random.randint(2, 100)
    img = np.random.rand(y_shape, x_shape)
    assert (
        img == img_tools.crop_center_2d(img, crop_x=x_shape, crop_y=y_shape)
    ).all()

    new_x_shape = random.randint(1, x_shape)
    new_y_shape = random.randint(1, y_shape)
    pad_img = img_tools.crop_center_2d(
        img, crop_x=new_x_shape, crop_y=new_y_shape
    )
    assert (new_y_shape, new_x_shape) == pad_img.shape


def test_pad_centre_2d():
    x_shape = random.randint(2, 100)
    y_shape = random.randint(2, 100)
    img = np.random.rand(y_shape, x_shape)
    assert (
        img == img_tools.pad_center_2d(img, x_size=x_shape, y_size=y_shape)
    ).all()

    new_x_shape = random.randint(x_shape, x_shape * 10)
    new_y_shape = random.randint(y_shape, y_shape * 10)
    pad_img = img_tools.pad_center_2d(
        img, x_size=new_x_shape, y_size=new_y_shape
    )
    assert (new_y_shape, new_x_shape) == pad_img.shape


@pytest.mark.parametrize("progress", [True, False])
def test_dataset_mean_std(progress):
    # checks that dataset_mean_std correctly computes the std/mean
    data = np.random.normal(100, 10, (10, 10, 10))

    mean, std = img_tools.dataset_mean_std(
        data, sampling_factor=2, show_progress=progress
    )
    # give it enough room for estimation error
    assert 90 < mean < 110
    assert 8 < std < 12


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tiled_var(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    # 10 planes, each with tiles containing 9 elements, and 16 tiles
    data = torch.arange(
        10 * 9 * 16, dtype=torch.float32, device=device
    ).reshape((10, 9, 16))
    data = torch.concat([data, data * 10000, data / 10000], dim=0)

    for i in range(data.shape[0]):
        batch = data[max(0, i - 4) : i + 1, :, :]
        var, mean = torch.var_mean(batch, dim=1, correction=0)
        torch_std, torch_mean = torch.std_mean(batch, dim=(0, 1), correction=1)

        our_mean, our_std = img_tools.batch_tiled_mean_std(mean, var, 9)
        assert our_mean.shape == (16,)
        assert our_std.shape == (16,)

        torch.allclose(our_mean, torch_mean)
        torch.allclose(our_std, torch_std)

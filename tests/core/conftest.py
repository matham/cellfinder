import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pooch
import pytest
import torch.backends.mps
from skimage.filters import gaussian

from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    download_models,
)
from cellfinder.core.tools.system import force_cpu


@pytest.fixture(scope="session", autouse=True)
def set_device_arm_macos_ci():
    """
    Ensure that the device is set to CPU when running on arm based macOS
    GitHub runners. This is to avoid the following error:
    https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773/5
    """
    if (
        os.getenv("GITHUB_ACTIONS") == "true"
        and torch.backends.mps.is_available()
    ):
        force_cpu()


@pytest.fixture(scope="session")
def no_free_cpus() -> int:
    """
    Set number of free CPUs so all available CPUs are used by the tests.
    """
    return 0


@pytest.fixture(scope="session")
def run_on_one_cpu_only() -> int:
    """
    Set number of free CPUs so tests can use exactly one CPU.
    """
    cpus = os.cpu_count()
    if cpus is not None:
        return cpus - 1
    else:
        raise ValueError("No CPUs available.")


@pytest.fixture(scope="session")
def download_default_model():
    """
    Check that the classification model is already downloaded
    at the beginning of a pytest session.
    """
    download_models("resnet50_tv", DEFAULT_DOWNLOAD_DIRECTORY)


@pytest.fixture(scope="session")
def synthetic_bright_spots() -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a synthetic signal array with grid of bright spots
    in a 3d numpy array to be used for cell detection testing.
    """
    shape = (100, 100, 100)

    signal_array = np.zeros(shape)
    signal_array[25, 25, 25] = 1
    signal_array[75, 25, 25] = 1
    signal_array[25, 75, 25] = 1
    signal_array[25, 25, 75] = 1
    signal_array[75, 75, 25] = 1
    signal_array[75, 25, 75] = 1
    signal_array[25, 75, 75] = 1
    signal_array[75, 75, 75] = 1

    # convert to 16-bit integer
    signal_array = (signal_array * 65535).astype(np.uint16)

    # blur a bit to roughly match the size of the cells in the sample data
    signal_array = gaussian(signal_array, sigma=2, preserve_range=True).astype(
        np.uint16
    )

    background_array = np.zeros_like(signal_array)

    return signal_array, background_array


@pytest.fixture
def test_data_registry():
    """
    Create a test data registry for BrainGlobe.

    Returns:
        pooch.Pooch: The test data registry object.

    """
    registry = pooch.create(
        pooch.os_cache("brainglobe_test_data"),
        base_url="https://gin.g-node.org/BrainGlobe/test-data/raw/master/",
        env="BRAINGLOBE_TEST_DATA_DIR",
    )
    registry.load_registry(Path(__file__).parent / "registry.txt")
    return registry


@pytest.fixture
def fetch_pooch_directory():
    """
    Fetches files from the Pooch registry that belong to a specific directory.

    Parameters:
        registry (pooch.Pooch): The Pooch registry object.
        directory_name (str):
            The remote relative path of the directory to fetch files from.
        processor (callable, optional):
            A function to process the fetched files. Defaults to None.
        downloader (callable, optional):
            A function to download the files. Defaults to None.
        progressbar (bool, optional):
            Whether to display a progress bar during the fetch.
            Defaults to False.

    Returns:
        str: The local absolute path to the fetched directory.
    """

    def _fetch_pooch_registry(
        registry: pooch.Pooch,
        directory_name: str,
        processor=None,
        downloader=None,
        progressbar=False,
    ):
        for name in registry.registry_files:
            if name.startswith(f"{directory_name}/"):
                registry.fetch(
                    name,
                    processor=processor,
                    downloader=downloader,
                    progressbar=progressbar,
                )

        return str(registry.abspath / directory_name)

    return _fetch_pooch_registry

from pathlib import Path


def test_bright_brain_data(test_data_registry, fetch_pooch_directory):
    bright_brain_relative_path = "cellfinder/bright_brain"
    bright_brain_root = fetch_pooch_directory(
        test_data_registry, bright_brain_relative_path
    )
    assert Path(
        bright_brain_root
    ).exists(), f"Unable to fetch {bright_brain_relative_path} "
    "from {test_data_registry.base_url}"

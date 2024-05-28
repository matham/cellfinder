import numpy as np
import torch
from pyinstrument import Profiler

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.tools.tools import get_max_possible_int_value

# Use random 16-bit integer data for signal plane
shape = (10000, 10000)
# use cpu/cuda
torch_device = "cpu"

signal_array_plane = np.random.randint(
    low=0, high=65536, size=shape, dtype=np.uint16
)


def setup_tile_filtering(plane: np.ndarray):
    max_value = get_max_possible_int_value(plane.dtype)
    clipping_value = max_value - 2
    thrsh_val = max_value - 1

    return clipping_value, thrsh_val


if __name__ == "__main__":
    clipping_value, threshold_value = setup_tile_filtering(signal_array_plane)
    # filtering needs float type, and add extra dim for z-stack
    signal_array_plane = (
        torch.from_numpy(signal_array_plane.astype(np.float32))
        .to(torch_device)
        .unsqueeze(0)
    )

    with torch.inference_mode(True):
        tile_processor = TileProcessor(
            plane_shape=shape,
            clipping_value=clipping_value,
            threshold_value=threshold_value,
            soma_diameter=16,
            log_sigma_size=0.2,
            n_sds_above_mean_thresh=10,
            torch_device=torch_device,
            dtype="float32",
            use_scipy=False,  # test pytorch implementation
        )

        profiler = Profiler()
        profiler.start()
        plane, tiles = tile_processor.get_tile_mask(signal_array_plane)
        profiler.stop()
        profiler.print(show_all=True)

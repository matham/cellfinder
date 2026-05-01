import numpy as np
from magicgui.widgets import ProgressBar
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from qtpy.QtCore import Signal

from cellfinder.core.detect.detect_debug import DetectionDebug
from cellfinder.core.main import main as cellfinder_run

from .detect_containers import (
    ClassificationInputs,
    DataInputs,
    DebugInputs,
    DetectionInputs,
    MiscInputs,
)


class MyWorkerSignals(WorkerBaseSignals):
    """
    Signals used by the Worker class below.
    """

    # Emits (label, max, value) for the progress bar
    update_progress_bar = Signal(str, int, int)


class Worker(WorkerBase):
    """
    Runs cellfinder in a separate thread, to prevent GUI blocking.

    Also handles callbacks between the worker thread and main napari GUI thread
    to update a progress bar.
    """

    def __init__(
        self,
        data_inputs: DataInputs,
        detection_inputs: DetectionInputs,
        classification_inputs: ClassificationInputs,
        misc_inputs: MiscInputs,
    ):
        super().__init__(SignalsClass=MyWorkerSignals)
        self.data_inputs = data_inputs
        self.detection_inputs = detection_inputs
        self.classification_inputs = classification_inputs
        self.misc_inputs = misc_inputs

    def connect_progress_bar_callback(self, progress_bar: ProgressBar):
        """
        Connects the progress bar to the work so that updates are shown on
        the bar.
        """

        def update_progress_bar(label: str, max: int, value: int):
            progress_bar.label = label
            progress_bar.max = max
            progress_bar.value = value

        self.update_progress_bar.connect(update_progress_bar)

    def work(self) -> list:
        if not self.detection_inputs.skip_detection:
            self.update_progress_bar.emit("Setting up detection...", 1, 0)

        def detect_callback(plane: int) -> None:
            if not self.detection_inputs.skip_detection:
                self.update_progress_bar.emit(
                    "Detecting cells",
                    self.data_inputs.nplanes,
                    plane + 1,
                )

        def detect_finished_callback(points: list) -> None:
            self.npoints_detected = len(points)
            if not self.classification_inputs.skip_classification:
                self.update_progress_bar.emit(
                    "Setting up classification...", 1, 0
                )

        def classify_callback(batch: int) -> None:
            if not self.classification_inputs.skip_classification:
                self.update_progress_bar.emit(
                    "Classifying cells",
                    # Default cellfinder-core batch size is 64.
                    # This seems to give a slight
                    # underestimate of the number of batches though,
                    # so allow for batch number to go over this
                    max(self.npoints_detected // 64 + 1, batch + 1),
                    batch + 1,
                )

        result = cellfinder_run(
            **self.data_inputs.as_core_arguments(),
            **self.detection_inputs.as_core_arguments(),
            **self.classification_inputs.as_core_arguments(),
            **self.misc_inputs.as_core_arguments(),
            detect_callback=detect_callback,
            classify_callback=classify_callback,
            detect_finished_callback=detect_finished_callback,
        )
        if not self.classification_inputs.skip_classification:
            self.update_progress_bar.emit("Finished classification", 1, 1)
        else:
            self.update_progress_bar.emit("Finished detection", 1, 1)
        return result


class DebugWorker(Worker):

    def __init__(self, *args, debug_inputs: DebugInputs, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_inputs = debug_inputs

    def work(self) -> DetectionDebug:
        self.update_progress_bar.emit("Setting up detection...", 1, 0)

        data = self.data_inputs
        detect = self.detection_inputs
        misc = self.misc_inputs
        debug = self.debug_inputs

        arr = data.signal_array
        start = misc.start_plane
        end = misc.end_plane

        plane_size = arr.shape[1], arr.shape[2]
        if DetectionDebug.needs_crop(debug.bottom_corner, debug.top_corner):
            bot_y, bot_x = debug.bottom_corner
            top_y, top_x = debug.top_corner
            bot_x = max(bot_x, 0)
            bot_y = max(bot_y, 0)
            plane_size = max(top_y - bot_y, 0), max(top_x - bot_x, 0)

        start = max(start, 0)
        end = min(len(arr) if end <= 0 else end, len(arr))
        n = max(end - start, 0)

        local_store = debug.local_store
        local_store_loader = None
        if debug.gen_dir:
            local_store_loader = DetectionDebug(
                (0, 0, 0), local_store=local_store
            )
            local_store = local_store / debug.gen_dir

        detect_debug = DetectionDebug(
            signal_shape=(n, plane_size[0], plane_size[1]),
            local_store=local_store,
            batch_size=detect.detection_batch_size,
            torch_device="cuda" if misc.use_gpu else "cpu",
            dtype=np.uint16,
            use_scipy=not misc.use_gpu,
            voxel_sizes=(
                data.voxel_size_z,
                data.voxel_size_y,
                data.voxel_size_x,
            ),
            soma_diameter=detect.soma_diameter,
            max_cluster_size=detect.max_cluster_size,
            ball_xy_size=detect.ball_xy_size,
            ball_z_size=detect.ball_z_size,
            ball_overlap_fraction=detect.ball_overlap_fraction,
            soma_spread_factor=detect.soma_spread_factor,
            n_free_cpus=misc.n_free_cpus,
            log_sigma_size=detect.log_sigma_size,
            n_sds_above_mean_thresh=detect.n_sds_above_mean_thresh,
            detect_centre_of_intensity=detect.detect_centre_of_intensity,
            split_ball_xy_size=detect.split_ball_xy_size,
            split_ball_z_size=detect.split_ball_z_size,
            split_ball_overlap_fraction=detect.split_ball_overlap_fraction,
            n_splitting_iter=detect.n_splitting_iter,
            n_sds_above_mean_tiled_thresh=detect.n_sds_above_mean_tiled_thresh,
            tiled_thresh_tile_size=detect.tiled_thresh_tile_size,
        )
        detect_debug.load_data(
            signal=arr,
            local_store_loader=local_store_loader,
            start_gen_from=debug.start_gen_from,
            end_gen_on=debug.end_gen_on,
            start_plane=start,
            end_plane=end,
            bottom_corner=debug.bottom_corner,
            top_corner=debug.top_corner,
        )

        def progress_callback(total_count: int, count: int, msg: str) -> None:
            self.update_progress_bar.emit(
                msg,
                total_count,
                count + 1,
            )

        detect_debug.run_filter(
            start_gen_from=debug.start_gen_from,
            end_gen_on=debug.end_gen_on,
            progress_callback=progress_callback,
        )

        self.update_progress_bar.emit("Finished detection", 1, 1)
        return detect_debug

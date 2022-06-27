import glob
import pathlib
from unittest.mock import patch

import napari
import numpy as np
import pytest
from PIL import Image

from cellfinder_napari.detect import detect_widget
from cellfinder_napari.detect.detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from cellfinder_napari.detect.thread_worker import Worker
from cellfinder_napari.sample_data import load_sample


@pytest.fixture
def get_detect_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = detect_widget()
    for layer in load_sample():
        viewer.add_layer(napari.layers.Image(layer[0], **layer[1]))
    viewer.window.add_dock_widget(widget)
    return widget


def test_add_detect_widget(get_detect_widget):
    """
    Smoke test to check that adding detection widget works
    """
    widget = get_detect_widget
    assert widget is not None


def test_detect_worker():
    """
    Smoke test to check that the detection worker runs
    """
    data = load_sample()
    signal = data[0][0]
    background = data[1][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=background),
        DetectionInputs(),
        ClassificationInputs(trained_model=None),
        MiscInputs(start_plane=0, end_plane=1),
    )
    worker.work()


@pytest.mark.parametrize(
    argnames="analyse_local",
    argvalues=[True, False],  # increase test coverage by covering both cases
)
def test_run_detect(get_detect_widget, analyse_local):
    """
    Test backend is called
    """
    with patch("cellfinder_napari.detect.detect.Worker") as worker:
        get_detect_widget.analyse_local.value = analyse_local
        get_detect_widget.call_button.clicked()
        assert worker.called


def test_reset_defaults(get_detect_widget):
    """Smoke test that restore defaults doesn't error."""
    get_detect_widget.reset_button.clicked()

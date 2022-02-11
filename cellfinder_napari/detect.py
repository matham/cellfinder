from math import ceil
from pathlib import Path
from typing import List

import napari
from cellfinder_core.classify.cube_generator import get_cube_depth_min_max
from magicgui import magicgui

from cellfinder_napari.detect_utils import add_layers, default_parameters, run
from cellfinder_napari.utils import brainglobe_logo

NETWORK_VOXEL_SIZES = [5, 1, 1]
CUBE_WIDTH = 50
CUBE_HEIGHT = 20
CUBE_DEPTH = 20

# If using ROI, how many extra planes to analyse
MIN_PLANES_ANALYSE = 0

def detect():
    DEFAULT_PARAMETERS = default_parameters()
    @magicgui(
        header=dict(
            widget_type="Label",
            label=f'<h1><img src="{brainglobe_logo}"width="100">cellfinder</h1>',
        ),
        detection_label=dict(
            widget_type="Label",
            label="<h3>Cell detection</h3>",
        ),
        data_options=dict(
            widget_type="Label",
            label="<b>Data:</b>",
        ),
        detection_options=dict(
            widget_type="Label",
            label="<b>Detection:</b>",
        ),
        classification_options=dict(
            widget_type="Label",
            label="<b>Classification:</b>",
        ),
        misc_options=dict(
            widget_type="Label",
            label="<b>Misc:</b>",
        ),
        voxel_size_z=dict(
            value=DEFAULT_PARAMETERS["voxel_size_z"],
            label="Voxel size (z)",
            step=0.1,
        ),
        voxel_size_y=dict(
            value=DEFAULT_PARAMETERS["voxel_size_y"],
            label="Voxel size (y)",
            step=0.1,
        ),
        voxel_size_x=dict(
            value=DEFAULT_PARAMETERS["voxel_size_x"],
            label="Voxel size (x)",
            step=0.1,
        ),
        Soma_diameter=dict(
            value=DEFAULT_PARAMETERS["Soma_diameter"], step=0.1
        ),
        ball_xy_size=dict(
            value=DEFAULT_PARAMETERS["ball_xy_size"], label="Ball filter (xy)"
        ),
        ball_z_size=dict(
            value=DEFAULT_PARAMETERS["ball_z_size"], label="Ball filter (z)"
        ),
        Ball_overlap=dict(value=DEFAULT_PARAMETERS["Ball_overlap"], step=0.1),
        Filter_width=dict(value=DEFAULT_PARAMETERS["Filter_width"], step=0.1),
        Threshold=dict(value=DEFAULT_PARAMETERS["Threshold"], step=0.1),
        Cell_spread=dict(value=DEFAULT_PARAMETERS["Cell_spread"], step=0.1),
        Max_cluster=dict(
            value=DEFAULT_PARAMETERS["Max_cluster"], min=0, max=10000000
        ),
        Trained_model=dict(value=DEFAULT_PARAMETERS["Trained_model"]),
        Start_plane=dict(
            value=DEFAULT_PARAMETERS["Start_plane"], min=0, max=100000
        ),
        End_plane=dict(
            value=DEFAULT_PARAMETERS["End_plane"], min=0, max=100000
        ),
        Number_of_free_cpus=dict(
            value=DEFAULT_PARAMETERS["Number_of_free_cpus"]
        ),
        Analyse_local=dict(
            value=DEFAULT_PARAMETERS["Analyse_local"], label="Analyse local"
        ),
        Debug=dict(value=DEFAULT_PARAMETERS["Debug"]),
        # Classification_batch_size=dict(max=4096),
        call_button=True,
        persist=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
    )
    def widget(
        header,
        detection_label,
        data_options,
        viewer: napari.Viewer,
        Signal_image: napari.layers.Image,
        Background_image: napari.layers.Image,
        voxel_size_z: float,
        voxel_size_y: float,
        voxel_size_x: float,
        detection_options,
        Soma_diameter: float,
        ball_xy_size: float,
        ball_z_size: float,
        Ball_overlap: float,
        Filter_width: float,
        Threshold: int,
        Cell_spread: float,
        Max_cluster: int,
        classification_options,
        Trained_model: Path,
        misc_options,
        Start_plane: int,
        End_plane: int,
        Number_of_free_cpus: int,
        Analyse_local: bool,
        Debug: bool,
        reset_button,
    ) -> List[napari.types.LayerDataTuple]:
        """

        Parameters
        ----------
        Signal_image : napari.layers.Image
             Image layer containing the labelled cells
        Background_image : napari.layers.Image
             Image layer without labelled cells
        voxel_size_z : float
            Size of your voxels in the axial dimension
        voxel_size_y : float
            Size of your voxels in the y direction (top to bottom)
        voxel_size_x : float
            Size of your voxels in the x direction (left to right)
        Soma_diameter : float
            The expected in-plane soma diameter (microns)
        ball_xy_size : float
            Elliptical morphological in-plane filter size (microns)
        ball_z_size : float
            Elliptical morphological axial filter size (microns)
        Ball_overlap : float
            Fraction of the morphological filter needed to be filled
            to retain a voxel
        Filter_width : float
            Laplacian of Gaussian filter width (as a fraction of soma diameter)
        Threshold : int
            Cell intensity threshold (as a multiple of noise above the mean)
        Cell_spread : float
            Cell spread factor (for splitting up cell clusters)
        Max_cluster : int
            Largest putative cell cluster (in cubic um) where splitting
            should be attempted
        Trained_model : Path
            Trained model file path (home directory (default) -> pretrained weights)
        Start_plane : int
            First plane to process (to process a subset of the data)
        End_plane : int
            Last plane to process (to process a subset of the data)
        Number_of_free_cpus : int
            How many CPU cores to leave free
        Analyse_local : bool
            Only analyse planes around the current position
        Debug : bool
            Increase logging
        reset_button :
            Reset parameters to default
        """

        if End_plane == 0:
            End_plane = len(Signal_image.data)

        voxel_sizes = (voxel_size_z, voxel_size_y, voxel_size_x)
        if Trained_model == Path.home():
            Trained_model = None

        if Analyse_local:
            current_plane = viewer.dims.current_step[0]

            # so a reasonable number of cells in the plane are detected
            planes_needed = MIN_PLANES_ANALYSE + int(
                ceil((CUBE_DEPTH * NETWORK_VOXEL_SIZES[0]) / voxel_size_z)
            )

            Start_plane, End_plane = get_cube_depth_min_max(
                current_plane, planes_needed
            )
            Start_plane = max(0, Start_plane)
            End_plane = min(len(Signal_image.data), End_plane)

        worker = run(
            Signal_image.data,
            Background_image.data,
            voxel_sizes,
            Soma_diameter,
            ball_xy_size,
            ball_z_size,
            Start_plane,
            End_plane,
            Ball_overlap,
            Filter_width,
            Threshold,
            Cell_spread,
            Max_cluster,
            Trained_model,
            Number_of_free_cpus,
            # Classification_batch_size,
        )
        worker.returned.connect(lambda points : add_layers(points, viewer=viewer))
        worker.start()

    widget.header.value = (
        "<p>Efficient cell detection in large images.</p>"
        '<p><a href="https://cellfinder.info" style="color:gray;">Website</a></p>'
        '<p><a href="https://docs.brainglobe.info/cellfinder/napari-plugin" style="color:gray;">Documentation</a></p>'
        '<p><a href="https://github.com/brainglobe/cellfinder-napari" style="color:gray;">Source</a></p>'
        '<p><a href="https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2" style="color:gray;">Citation</a></p>'
        "<p><small>For help, hover the cursor over each parameter.</small>"
    )
    widget.header.native.setOpenExternalLinks(True)

    @widget.reset_button.changed.connect
    def restore_defaults(event=None):
        for name, value in DEFAULT_PARAMETERS.items():
            getattr(widget, name).value = value

    return widget

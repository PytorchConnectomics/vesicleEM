"""
PyVista Visualization Generation Script

Creates 3D visualizations of neuron and vesicle meshes using PyVista.
Supports interactive viewing and screenshot generation with customizable camera positions.
"""

import os
import csv
import trimesh
import numpy as np
import pyvista as pv
import argparse
from pathlib import Path

# TODO: logging with tqdm
# TODO: generalize beyond obj

def load_offsets(csv_path):
    """
    Loads offsets from a CSV file into a dictionary.

    The CSV file should have columns 'name', 'x', 'y', and 'z'.
    The 'name' column is used as the key in the dictionary, and the
    'x', 'y', and 'z' columns are used to create a NumPy array of
    offsets.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary where the keys are the names from the CSV
            file and the values are NumPy arrays representing the
            offsets.
    """
    offsets = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            offsets[name] = np.array([int(row['x']), int(row['y']), int(row['z'])])
    return offsets

def load_trimesh_objs(folder):
    """
    Loads all .obj files from a given folder into a dictionary of trimesh objects.

    Args:
        folder (str): The path to the folder containing the .obj files.

    Returns:
        dict: A dictionary where keys are the names of the .obj files (without extension)
                and values are the corresponding trimesh objects.
                Returns an empty dictionary if no .obj files are found.
    """
    meshes = {}
    for filename in os.listdir(folder):
        if filename.endswith('.obj'):
            name = os.path.splitext(filename)[0]
            path = os.path.join(folder, filename)
            meshes[name] = trimesh.load(path, process=False)
    return meshes

def load_vesicle_trimesh_objs(folder):
    """
    Loads vesicle meshes from OBJ files in a folder, grouping them by neuron ID.

    Args:
        folder (str): Path to the folder containing the OBJ files.  Filenames
            are expected to start with the neuron ID, followed by an underscore.

    Returns:
        dict: A dictionary where keys are neuron IDs and values are lists of
            trimesh.Trimesh objects representing the vesicles for that neuron.
            Returns an empty dictionary if no OBJ files are found.
    """
    vesicles = {}
    for filename in os.listdir(folder):
        if filename.endswith('.obj'):
            neuron_id = filename.split('_')[0]
            path = os.path.join(folder, filename)
            mesh = trimesh.load(path, process=False)
            vesicles.setdefault(neuron_id, []).append(mesh)
    return vesicles

def trimesh_to_pyvista(mesh):
    """
    Convert a trimesh mesh to a PyVista PolyData object, handling vertex colors.

        This function takes a trimesh.Trimesh object as input and converts it into a
        PyVista PolyData object suitable for visualization and further processing within
        the PyVista ecosystem.  It correctly handles the mesh's vertices and faces,
        and crucially, it also transfers vertex color information if it is present in
        the trimesh object.

        Args:
            mesh (trimesh.Trimesh): The input trimesh.Trimesh object to convert.  This
                object is expected to contain vertex and face data, and optionally,
                vertex color data accessible via `mesh.visual.vertex_colors`.

        Returns:
            pyvista.PolyData: A PyVista PolyData object representing the converted mesh.
                The PolyData object contains the mesh's vertices and faces. If the input
                trimesh object has vertex colors, these are also included in the
                PolyData as a point data array named 'Colors'.

        Raises:
            AttributeError: If the input `mesh` does not have the expected attributes
                (e.g., `vertices`, `faces`, or `visual`).
            TypeError: If the vertex color data is not of the expected type (e.g., not a
                NumPy array) or if the face data is not convertible to int64.
            ValueError: If the vertex color data has an unexpected shape (e.g., not an
                Nx3 or Nx4 array).  If an Nx4 array is provided, the alpha channel is
                discarded.
    """
    faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),
        mesh.faces
    ]).astype(np.int64)
    pd_mesh = pv.PolyData(mesh.vertices, faces=faces)

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        pd_mesh.point_data['Colors'] = colors
    return pd_mesh

def render_scene(neuron_mesh_dir, vesicle_mesh_dir, offsets_csv, show_plotter=True, camera_position=None, screenshot_path='scene.png'):
    """
    Renders a scene composed of neuron and vesicle meshes using PyVista.
    This function loads neuron and vesicle meshes from specified directories,
    applies positional offsets, and visualizes them in a PyVista plotter.
    It supports interactive display or saving a screenshot of the scene.
    Args:
        neuron_mesh_dir (str): Path to the directory containing neuron mesh files in OBJ format.
                                    The filenames (without extension) are used as neuron IDs.
        vesicle_mesh_dir (str): Path to the directory containing vesicle mesh files in OBJ format.
                                    Vesicles are expected to be organized in subdirectories named after
                                    the neuron ID they belong to.
        offsets_csv (str): Path to a CSV file containing positional offsets for each neuron.
                                The CSV should have columns 'id', 'x', 'y', and 'z'.
                                If an ID is not found in the CSV, a default offset of [0, 0, 0] is used.
        show_plotter (bool, optional): If True, displays the scene in an interactive PyVista plotter window.
                                        If False, renders the scene off-screen and saves a screenshot.
                                        Defaults to True.
        camera_position (list or str, optional):  The camera position to use for the plot.  Can be a list of
                                                    three points defining the camera position, focal point,
                                                    and view-up direction, or a string such as 'xy', 'xz', etc.
                                                    Defaults to None, which uses the PyVista default camera position.
        screenshot_path (str, optional): Path to save the screenshot of the scene if `show_plotter` is False.
                                            Defaults to 'scene.png'.
    Returns:
        None
    Raises:
        FileNotFoundError: If the specified neuron or vesicle mesh directories do not exist.
        ValueError: If the offsets CSV file is not properly formatted.
    Notes:
        - The function assumes that neuron mesh files have the .obj extension.
        - Vesicle mesh directories should contain .obj files directly.
        - The offsets CSV file should have a header row.
        - The function uses `trimesh` for loading meshes and `pyvista` for rendering.
        - If `show_plotter` is False, the function runs in off-screen mode, which requires a properly configured
            environment for off-screen rendering.
    Examples:
        Render the scene with default settings:
        ```python
        render_scene('neurons/', 'vesicles/', 'offsets.csv')
        ```
        Render the scene and save a screenshot:
        ```python
        render_scene('neurons/', 'vesicles/', 'offsets.csv', show_plotter=False, screenshot_path='my_scene.png')
        ```
        Render the scene with a specific camera position:
        ```python
        camera_pos = [(10, 10, 10), (0, 0, 0), (0, 1, 0)]  # Camera position, focal point, view-up
        render_scene('neurons/', 'vesicles/', 'offsets.csv', camera_position=camera_pos)
        ```
    """
    
    offsets = load_offsets(offsets_csv)

    neurons = load_trimesh_objs(neuron_mesh_dir)

    vesicles = load_vesicle_trimesh_objs(vesicle_mesh_dir)

    plotter = pv.Plotter(off_screen=not show_plotter)

    for name, mesh in neurons.items():
        offset = offsets.get(name, np.array([0, 0, 0]))
        mesh.apply_translation(offset)
        pd_mesh = trimesh_to_pyvista(mesh)
        plotter.add_mesh(pd_mesh, name=f'neuron_{name}', show_scalar_bar=False)

    for neuron_id, vesicle_list in vesicles.items():
        offset = offsets.get(neuron_id, np.array([0, 0, 0]))
        for vmesh in vesicle_list:
            vmesh.apply_translation(offset)
            pd_vmesh = trimesh_to_pyvista(vmesh)
            plotter.add_mesh(pd_vmesh, name=f'vesicle_{neuron_id}', show_scalar_bar=False)

    if camera_position:
        plotter.camera_position = camera_position

    if show_plotter:
        plotter.show()
    else:
        img = plotter.show(screenshot=screenshot_path)

def main():
    """Main function to handle command-line arguments and render PyVista scene."""
    parser = argparse.ArgumentParser(
        description="Render 3D scene of neuron and vesicle meshes using PyVista",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "neuron_mesh_dir",
        type=str,
        help="Directory containing neuron mesh files (.obj format)"
    )
    
    parser.add_argument(
        "vesicle_mesh_dir",
        type=str,
        help="Directory containing vesicle mesh files (.obj format)"
    )
    
    parser.add_argument(
        "offsets_csv",
        type=str,
        help="CSV file with neuron offsets (columns: name, x, y, z)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive display (save screenshot only)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="scene.png",
        help="Screenshot output path (used when --no-interactive is set)"
    )
    
    parser.add_argument(
        "--camera-position",
        type=str,
        default=None,
        choices=["xy", "xz", "yz", "-xy", "-xz", "-yz"],
        help="Predefined camera position"
    )
    
    parser.add_argument(
        "--camera-custom",
        type=float,
        nargs=9,
        metavar=("CX", "CY", "CZ", "FX", "FY", "FZ", "UX", "UY", "UZ"),
        help="Custom camera position: camera_pos focal_point view_up (9 values)"
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    if not Path(args.neuron_mesh_dir).is_dir():
        print(f"Error: Neuron mesh directory not found: {args.neuron_mesh_dir}")
        return 1
    
    if not Path(args.vesicle_mesh_dir).is_dir():
        print(f"Error: Vesicle mesh directory not found: {args.vesicle_mesh_dir}")
        return 1
    
    if not Path(args.offsets_csv).is_file():
        print(f"Error: Offsets CSV file not found: {args.offsets_csv}")
        return 1
    
    # Parse camera position
    camera_position = None
    if args.camera_custom:
        camera_position = [
            args.camera_custom[0:3],  # camera position
            args.camera_custom[3:6],  # focal point
            args.camera_custom[6:9]   # view up
        ]
    elif args.camera_position:
        camera_position = args.camera_position
    
    try:
        render_scene(
            neuron_mesh_dir=args.neuron_mesh_dir,
            vesicle_mesh_dir=args.vesicle_mesh_dir,
            offsets_csv=args.offsets_csv,
            show_plotter=not args.no_interactive,
            camera_position=camera_position,
            screenshot_path=args.output
        )
        
        if args.no_interactive:
            print(f"Scene rendered and saved to: {args.output}")
        else:
            print("Interactive PyVista scene displayed!")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
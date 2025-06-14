#!/usr/bin/env python3
"""
Vesicle Mesh Generation Script

Generates a 3D mesh of vesicles from a parquet file containing vesicle data.
Each vesicle is represented as a sphere with customizable resolution and coloring.
"""

import numpy as np
import polars as pl
import trimesh
import argparse
from pathlib import Path
from matplotlib import cm
from matplotlib import colors as mcolors


def vesicles_to_mesh(
    parquet_path,
    output_path,
    output_format="obj",
    sphere_resolution=4,
    color_by=None,
    colormap="viridis",
    verbose=True
):
    """
    Generates a mesh of vesicles from a parquet file and saves it.
    
    Parameters:
    - parquet_path (str or Path): Path to .parquet with 'x', 'y', 'z', 'radius' and optional data columns.
    - output_path (str or Path): Path to save the output mesh file.
    - output_format (str): Desired output format (e.g., "obj", "ply", "stl").
    - sphere_resolution (int): Level of icosphere detail per vesicle.
    - color_by (str or list of str, optional): Column(s) in the Parquet to base vertex color on.
    - colormap (str or callable): Matplotlib colormap name or custom [0,1] -> RGBA callable.
    - verbose (bool): If True, print progress.

    Returns:
    - None. The mesh is saved to the specified output_path.
    """
    parquet_path = Path(parquet_path)
    df = pl.read_parquet(parquet_path)

    required_cols = {"x", "y", "z", "radius"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Parquet must contain at least these columns: {required_cols}")

    if verbose:
        print(f"Loaded {len(df)} vesicles from {parquet_path.name}")

    coords = df.select(["x", "y", "z"]).to_numpy()
    radii = df.select("radius").to_numpy().flatten()

    colors = compute_vertex_colors(df, color_by=color_by, colormap=colormap)

    sphere_meshes = []
    for i, (center, r, color) in enumerate(zip(coords, radii, colors)):
        sphere = trimesh.creation.icosphere(subdivisions=sphere_resolution, radius=r)
        sphere.apply_translation(center)
        sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
        sphere_meshes.append(sphere)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(coords)} vesicles...")

    if not sphere_meshes:
        print("Warning: No vesicles processed, resulting mesh will be empty.")
        combined_mesh = trimesh.Trimesh()
    else:
        combined_mesh = trimesh.util.concatenate(sphere_meshes)

    if verbose:
        print(f"Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")

    try:
        supported_formats = set(trimesh.exchange.export.mesh_formats())
    except AttributeError:
        supported_formats = {"obj", "stl", "ply", "glb", "gltf", "collada", "dae", "off", "xyz", "json", "dict", "dict64", "msgpack"}
        if verbose:
            print("Warning: Could not dynamically fetch supported formats from trimesh. Using a predefined list.")

    if output_format.lower() not in supported_formats:
        print(f"Error: Invalid output format '{output_format}'.")
        print(f"Supported formats are: {', '.join(sorted(list(supported_formats)))}")
        return

    if combined_mesh.is_empty:
        print(f"Warning: The generated mesh is empty. Skipping export to {output_path}.")
        return

    print(f"Saving {output_format.upper()} file to {output_path}...")
    try:
        combined_mesh.export(output_path, file_type=output_format.lower())
        print(f"{output_format.upper()} Export Complete!")
    except Exception as e:
        print(f"Error exporting mesh to '{output_path}' as {output_format.upper()}: {e}")
        print("Please ensure the output path is valid, writable, and the format is correctly supported by your trimesh installation.")


def compute_vertex_colors(df, color_by=None, colormap="viridis"):
    """
    Computes per-row RGB colors based on one or more DataFrame columns.

    Parameters:
    - df (pl.DataFrame): DataFrame with vesicle data.
    - color_by (str or list of str): Column(s) to derive color from.
    - colormap (str or callable): Matplotlib colormap name or function.

    Returns:
    - colors (np.ndarray): (N, 4) array of RGBA values (uint8).
    """
    if color_by is None:
        return np.tile([200, 200, 200, 255], (df.height, 1))

    if isinstance(color_by, str):
        values = df[color_by].to_numpy()
    elif isinstance(color_by, list):
        values = df.select(color_by).to_numpy().sum(axis=1)
    else:
        raise ValueError("color_by must be a string or list of strings")

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
    rgba = cmap(norm(values))
    rgb255 = (rgba * 255).astype(np.uint8)
    return rgb255


def main():
    """Main function to handle command-line arguments and run vesicle mesh generation."""
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from vesicle data in parquet format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to parquet file containing vesicle data (must have x, y, z, radius columns)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="vesicles.obj",
        help="Output mesh file path"
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="obj",
        choices=["obj", "stl", "ply", "glb", "gltf", "collada", "dae", "off"],
        help="Output mesh format"
    )
    
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=4,
        help="Sphere resolution (icosphere subdivisions)"
    )
    
    parser.add_argument(
        "-c", "--color-by",
        type=str,
        default=None,
        help="Column name to base vertex colors on (e.g., 'type', 'volume')"
    )
    
    parser.add_argument(
        "-m", "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for coloring"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        vesicles_to_mesh(
            parquet_path=args.parquet_path,
            output_path=args.output,
            output_format=args.format,
            sphere_resolution=args.resolution,
            color_by=args.color_by,
            colormap=args.colormap,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\nVesicle mesh generation complete!")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
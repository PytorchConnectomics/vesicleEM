"""
HTML Viewer Generation Script

Generates an HTML visualization from vesicle data, neuron models, and color maps.
Creates a complete viewer package with all necessary files copied to output directory.
"""

import os
import json
import shutil
import pandas as pd
import pyarrow.parquet as pq
import argparse
from pathlib import Path

TEMPLATE_HTML_PATH = "vesicle_template.html"

def convert_parquet_to_json(parquet_path, output_path):
    """
        Convert Parquet to JSON

        This function converts a Parquet file to a JSON file.

        Args:
            parquet_path (str): The path to the input Parquet file.
            output_path (str): The path to the output JSON file.

        Returns:
            None
    """
    df = pd.read_parquet(parquet_path)
    vesicles = df.to_dict(orient="records")
    with open(output_path, 'w') as f:
        json.dump(vesicles, f)
    print(f"Converted vesicle data to JSON: {output_path}")

def parse_offset_csv(csv_path):
    """
    Parses a CSV file to extract neuron offsets.

    Args:
        csv_path (str): The path to the CSV file containing neuron offset data.
            The CSV should have the format: neuron, z_min, z_max, y_min, y_max, x_min, x_max, ... (other columns are ignored)
            where:
                - neuron (str): The name of the neuron.
                - z_min (int): The minimum Z coordinate.
                - z_max (int): The maximum Z coordinate.
                - y_min (int): The minimum Y coordinate.
                - y_max (int): The maximum Y coordinate.
                - x_min (int): The minimum X coordinate.
                - x_max (int): The maximum X coordinate.

    Returns:
        dict: A dictionary where keys are neuron names (str) and values are dictionaries
            containing the x, y, and z offsets (int).  For example:
            {'neuron_A': {'x': 10, 'y': 20, 'z': 30}, 'neuron_B': {'x': 40, 'y': 50, 'z': 60}}
            Note: Applies a +4000 offset to the y coordinate of neuron "SHL17".
    """
    offsets = {}
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            neuron, z_min, z_max, y_min, y_max, x_min, x_max = parts
            x_offset = int(x_min)
            y_offset = int(y_min)
            z_offset = int(z_min)
            if neuron.strip() == "SHL17":
                y_offset += 4000
            offsets[neuron.strip()] = {"x": x_offset, "y": y_offset, "z": z_offset}
    return offsets

def generate_html(
    vesicle_parquet_path: str,
    offset_csv_path: str,
    neuron_glb_dir: str,
    vesicle_color_map_path: str,
    output_dir: str = "vesicle_viewer_output",
    template_path: str = None,
    verbose: bool = True
):
    """
    Generates an HTML file for visualizing vesicles and neurons.
    This function takes paths to vesicle data, neuron models, and a color map,
    and generates an HTML file that can be used to visualize the data in a web browser.
    It also copies the vesicle data, neuron models, and color map to the output directory.
    Args:
        vesicle_parquet_path (str): Path to the Parquet file containing vesicle data.
        offset_csv_path (str): Path to the CSV file containing offset data (currently unused).
        neuron_glb_dir (str): Path to the directory containing neuron GLB files.
        vesicle_color_map_path (str): Path to the JSON file containing the vesicle color map.
        output_dir (str, optional): Path to the output directory. Defaults to "vesicle_viewer_output".
        template_path (str, optional): Path to HTML template file. Defaults to "vesicle_template.html".
        verbose (bool): Print status messages.
    Returns:
        None: This function does not return any value. It generates files in the specified output directory.
    """
    if template_path:
        template_file_path = template_path
    else:
        template_file_path = TEMPLATE_HTML_PATH
    
    if not Path(template_file_path).exists():
        raise FileNotFoundError(f"HTML template file not found: {template_file_path}")
    
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    neuron_output_dir = output_dir / "neurons"

    if verbose:
        print(f"Creating output directories at: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    neuron_output_dir.mkdir(parents=True, exist_ok=True)

    vesicle_json_path = data_dir / "vesicles.json"
    convert_parquet_to_json(vesicle_parquet_path, vesicle_json_path)

    color_map_output_path = data_dir / "colormap.json"
    shutil.copy(vesicle_color_map_path, color_map_output_path)
    if verbose:
        print(f"Custom color map copied to: {color_map_output_path}")

    neuron_files = list(Path(neuron_glb_dir).glob("*.glb"))
    for file in neuron_files:
        shutil.copy(file, neuron_output_dir / file.name)
    if verbose:
        print(f"Copied {len(neuron_files)} neuron GLB files to: {neuron_output_dir}")

    with open(template_file_path, 'r') as template_file:
        html_content = template_file.read()

    html_output_path = output_dir / "index.html"
    with open(html_output_path, 'w') as out_html:
        out_html.write(html_content)
    if verbose:
        print(f"Viewer HTML created at: {html_output_path}")


def main():
    """Main function to handle command-line arguments and run HTML generation."""
    parser = argparse.ArgumentParser(
        description="Generate HTML viewer for vesicle and neuron visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "vesicle_parquet",
        type=str,
        help="Path to vesicle data parquet file"
    )
    
    parser.add_argument(
        "offset_csv",
        type=str,
        help="Path to CSV file containing neuron offset data"
    )
    
    parser.add_argument(
        "neuron_glb_dir",
        type=str,
        help="Directory containing neuron GLB files"
    )
    
    parser.add_argument(
        "color_map",
        type=str,
        help="Path to vesicle color map JSON file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="vesicle_viewer_output",
        help="Output directory for generated viewer"
    )
    
    parser.add_argument(
        "-t", "--template",
        type=str,
        default=None,
        help="Path to HTML template file (defaults to vesicle_template.html)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        generate_html(
            vesicle_parquet_path=args.vesicle_parquet,
            offset_csv_path=args.offset_csv,
            neuron_glb_dir=args.neuron_glb_dir,
            vesicle_color_map_path=args.color_map,
            output_dir=args.output,
            template_path=args.template,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\nHTML viewer generation complete!")
            print(f"Open {Path(args.output) / 'index.html'} in your browser to view the visualization.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
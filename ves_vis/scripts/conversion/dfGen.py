# TODO: example txt files in readme

from pathlib import Path
import re
import polars as pl
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import argparse

def extract_vesicle_data(
    input_path,
    output_path="vesicle_com_data.parquet",
    types_dir=None,
    compute_neighbors=False,
    neighbor_radius_nm=500.0,
    voxel_dims_nm=(30, 8, 8),
    verbose=True
):
    """
    Parses vesicle mapping file(s) to extract center-of-mass (COM) and metadata,
    swaps X and Z coordinates, optionally joins type labels from a directory,
    and optionally computes neighbor densities within a specified radius.

    Parameters:
    - input_path (str or Path): File or directory with *_mapping.txt files.
    - output_path (str): Where to save result (.parquet, .csv, .json).
    - types_dir (str or Path or None): Directory with *_label.txt files (optional).
    - compute_neighbors (bool): Whether to compute neighbor densities (default: False).
    - neighbor_radius_nm (float): Radius in nanometers for KDTree neighbor ball (default: 500).
    - voxel_dims_nm (tuple or list): Physical dimensions (x,y,z) of a voxel in nanometers.
                                     Defaults to (30, 8, 8). Note that the function
                                     internally swaps X and Z, so for KDTree calculations,
                                     the order applied to coordinates (z_swapped, y, x_swapped)
                                     will effectively be (voxel_x, voxel_y, voxel_z).
    - verbose (bool): Print status and preview (default: True).

    Assumes voxel physical dimensions are (x=30nm, y=8nm, z=8nm) by default.
    These may vary by dataset. Ensure `voxel_dims_nm` is set correctly if your data differs.

    Returns:
    - df (pl.DataFrame): Extracted DataFrame with optional neighbor info.
    """
    input_path = Path(input_path).resolve()
    output_file = Path(output_path).resolve()
    all_files = []

    if verbose:
        print(f"Starting vesicle data extraction from: {input_path}")

    if input_path.is_file() and input_path.name.endswith("_mapping.txt"):
        all_files = [input_path]
    elif input_path.is_dir():
        all_files = list(input_path.glob("*_mapping.txt"))
    else:
        raise ValueError("Input path must be a *_mapping.txt file or a directory of them.")

    if not all_files:
        raise ValueError(f"No *_mapping.txt files found in {input_path}")
    
    if verbose:
        print(f"Found {len(all_files)} mapping files to process.")

    records = []
    pattern = re.compile(
        r'\((\d+\.?\d*), (\d+\.?\d*), (\d+\.?\d*)\): \(\'(\w+)\', (\w+_\d+), (\d+), (\d+\.?\d*(?:e[-+]?\d+)?)'
    )

    file_iterator = tqdm(all_files, desc="Processing mapping files", unit="file") if verbose else all_files
    for file_path in file_iterator:
        if verbose:
            print(f"Processing mapping file: {file_path.name}")
        vesicle_type = "lv" if "_lv_" in file_path.name else "sv"
        sample_id = file_path.stem.split('_')[0]

        with open(file_path, 'r') as f:
            content = f.read()

        matches = pattern.findall(content)

        for match in matches:
            x, y, z = float(match[0]), float(match[1]), float(match[2])
            _, v_id, volume, radius = match[3], match[4], int(match[5]), float(match[6])
            records.append({
                "sample_id": sample_id,
                "vesicle_type": vesicle_type,
                "vesicle_id": v_id,
                "x": x,
                "y": y,
                "z": z,
                "volume": volume,
                "radius": radius
            })

    if not records:
        raise ValueError("No valid vesicle entries found after parsing mapping files.")
    
    if verbose:
        print(f"Extracted {len(records)} vesicle records.")

    df = pl.DataFrame(records)

    if verbose:
        print("Swapping X and Z coordinates.")
    df = df.rename({"x": "temp_x", "z": "x"})
    df = df.rename({"temp_x": "z"})

    if types_dir:
        types_dir = Path(types_dir).resolve()
        if verbose:
            print(f"Processing type labels from directory: {types_dir}")
        label_files = list(types_dir.glob("*_lv_label.txt")) + list(types_dir.glob("*_sv_label.txt"))
        
        if not label_files:
            print(f"Warning: No label files found in {types_dir}. Skipping type assignment.")
        else:
            if verbose:
                print(f"Found {len(label_files)} label files.")
            type_records = []
            type_pattern = re.compile(r'\((\d+):(\d+)\)')

            label_iterator = tqdm(label_files, desc="Processing label files", unit="file") if verbose else label_files
            for file_path in label_iterator:
                if verbose:
                    print(f"Processing label file: {file_path.name}")
                vesicle_prefix = "lv" if "_lv_" in file_path.name else "sv"
                sample_id = file_path.stem.split('_')[0]

                with open(file_path, 'r') as f:
                    content = f.read()

                matches = type_pattern.findall(content)

                for vesicle_num, type_val in matches:
                    vesicle_id = f"{vesicle_prefix}_{vesicle_num}"
                    type_records.append({
                        "sample_id": sample_id,
                        "vesicle_id": vesicle_id,
                        "type": int(type_val)
                    })
            
            if type_records:
                df_types = pl.DataFrame(type_records)
                df = df.join(df_types, on=["sample_id", "vesicle_id"], how="left")
                df = df.with_columns(pl.col("type").fill_null(0))
                if verbose:
                    print("Joined type labels with main data. Filled missing types with 0.")
            else:
                if verbose:
                    print("No type records extracted from label files. Skipping type join.")
                df = df.with_columns(pl.lit(0).alias("type"))


    if compute_neighbors:
        if verbose:
            print(f"Computing neighbor counts within {neighbor_radius_nm}nm...")
            print(f"Using voxel dimensions (X, Y, Z) for scaling: {voxel_dims_nm} nm")

        df = df.with_row_count("_row_id")
        
        voxel_dims_np = np.array(voxel_dims_nm)
        
        all_counts = {}
        
        unique_sample_ids = df["sample_id"].unique().to_list()
        sample_iterator = tqdm(unique_sample_ids, desc="Computing neighbors per sample", unit="sample") if verbose else unique_sample_ids

        for sample_id in sample_iterator:
            if verbose:
                print(f"Processing sample for neighbor computation: {sample_id}")
            sample_df = df.filter(pl.col("sample_id") == sample_id)
            
            coords_for_kdtree = sample_df.select(["z", "y", "x"]).to_numpy() * voxel_dims_np
            
            row_ids = sample_df["_row_id"].to_numpy()

            if len(coords_for_kdtree) == 0:
                if verbose:
                    print(f"No coordinates for sample {sample_id}, skipping KDTree.")
                continue

            tree = KDTree(coords_for_kdtree)
            neighbors = tree.query_ball_point(coords_for_kdtree, r=neighbor_radius_nm)
            
            for i, neighbor_indices in enumerate(neighbors):
                all_counts[row_ids[i]] = len(neighbor_indices) - 1

        counts_list = [all_counts.get(i, 0) for i in range(df.height)]
        df = df.with_columns(pl.Series("neighbors_within_radius", counts_list))
        df = df.drop("_row_id")
        if verbose:
            print("Finished computing neighbor counts.")

    ext = output_file.suffix.lower()
    if ext == ".csv":
        df.write_csv(output_file)
    elif ext == ".json":
        df.write_json(output_file, row_oriented=True)
    else:
        if ext != ".parquet":
            output_file = output_file.with_suffix(".parquet")
            if verbose:
                print(f"Unsupported output extension '{ext}'. Saving as Parquet: {output_file}")
        df.write_parquet(output_file)

    if verbose:
        print(f"Data successfully saved to: {output_file}")
        print("DataFrame head:")
        print(df.head())
        print("Extraction process complete.")

    return df

def main():
    """Main function to handle command-line arguments and run the extraction."""
    parser = argparse.ArgumentParser(
        description="Extract vesicle data from mapping files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a *_mapping.txt file or directory containing such files"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="vesicle_com_data.parquet",
        help="Output file path (.parquet, .csv, or .json)"
    )
    
    parser.add_argument(
        "-t", "--types-dir",
        type=str,
        default=None,
        help="Directory containing *_label.txt files for type labels"
    )
    
    parser.add_argument(
        "-n", "--compute-neighbors",
        action="store_true",
        help="Compute neighbor densities within specified radius"
    )
    
    parser.add_argument(
        "-r", "--neighbor-radius",
        type=float,
        default=500.0,
        help="Radius in nanometers for neighbor search"
    )
    
    parser.add_argument(
        "-d", "--voxel-dims",
        type=float,
        nargs=3,
        default=[30.0, 8.0, 8.0],
        metavar=("X", "Y", "Z"),
        help="Voxel dimensions in nanometers (X Y Z)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        df = extract_vesicle_data(
            input_path=args.input_path,
            output_path=args.output,
            types_dir=args.types_dir,
            compute_neighbors=args.compute_neighbors,
            neighbor_radius_nm=args.neighbor_radius,
            voxel_dims_nm=tuple(args.voxel_dims),
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\nProcessing complete! Final DataFrame shape: {df.shape}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

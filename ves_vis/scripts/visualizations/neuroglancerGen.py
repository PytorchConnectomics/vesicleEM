"""
Neuroglancer Viewer Generation Script

Launches a Neuroglancer viewer to visualize neuron and vesicle data from HDF5 files.
Creates segmentation layers with appropriate offsets for spatial alignment.
"""

import os
import h5py
import gc
import csv
import socket
import numpy as np
import neuroglancer
import logging
import argparse
from contextlib import closing
from pathlib import Path

# TODO: improve logging via tqdm
# TODO improve name regex to not require hardcoded name structure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def append_seg_layer(viewer_txn, name, data, res, offset):
    """Appends a segmentation layer to a Neuroglancer viewer transaction.

        Args:
            viewer_txn: The Neuroglancer viewer transaction to append the layer to.
            name: The name of the layer.
            data: The segmentation data.
            res: The resolution of the data.
            offset: The offset of the data.
        """
    if data is not None:
        viewer_txn.layers.append(
            name=name,
            layer=ng_seg_layer(data, res, offset)
        )

def find_free_port():
    """Finds a free port on the system.

        This function creates a socket, binds it to a random port, and then returns the port number.
        This can be useful for applications that need to listen on a port but don't want to hardcode a specific port number.

        Returns:
            int: A free port number on the system.
        """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def load_offsets_csv(csv_path):
    """Loads offsets from a CSV file into a dictionary.

        The CSV file should have columns 'name', 'z', 'y', and 'x'.
        The 'name' column is the key in the dictionary, and the 'z', 'y', and 'x' columns are the values (as a list of integers).

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            dict: A dictionary where keys are names and values are lists of [z, y, x] offsets.
                  Returns None if the file is not found or if there's a missing column.
        """
    offsets = {}
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                offsets[name] = [int(row['z']), int(row['y']), int(row['x'])]
    except FileNotFoundError:
        logging.error(f"Offset file not found: {csv_path}")
        return None
    except KeyError as e:
        logging.error(f"Missing column in offset file: {e}")
        return None
    return offsets

def load_h5_volume(path, key):
    """Loads a volume from an HDF5 file.

        Args:
            path (str): The path to the HDF5 file.
            key (str): The key of the volume to load.

        Returns:
            np.ndarray: The volume as a NumPy array, or None if the file or key is not found.
        """
    try:
        with h5py.File(path, 'r') as f:
            return np.array(f[key])
    except FileNotFoundError:
        logging.error(f"HDF5 file not found: {path}")
        return None
    except KeyError:
        logging.error(f"Key not found in HDF5 file: {key} in {path}")
        return None

def ng_seg_layer(data, res, offset):
    """Generates a Neuroglancer segmentation layer from a data array.

        Args:
            data (np.ndarray): The segmentation data as a NumPy array.  Should be convertible to np.uint32.
            res (list or tuple): A list or tuple of three numbers representing the resolution (in nm) of the data in z, y, and x dimensions.
            offset (list or tuple): A list or tuple of three numbers representing the offset of the data in z, y, and x dimensions.

        Returns:
            neuroglancer.LocalVolume: A Neuroglancer LocalVolume object configured for segmentation.
        """
    return neuroglancer.LocalVolume(
        data.astype(np.uint32),
        dimensions=neuroglancer.CoordinateSpace(
            names=["z", "y", "x"],
            units=["nm", "nm", "nm"],
            scales=res
        ),
        voxel_offset=offset,
        volume_type='segmentation'
    )

def launch_neuroglancer(neuron_h5_dir, vesicle_h5_path, offset_csv_path, voxel_resolution=(30, 64, 64), bind_address='localhost', verbose=True):
    """
    Launches a Neuroglancer viewer to visualize neuron and vesicle data.

    Args:
        neuron_h5_dir (str): Path to the directory containing neuron HDF5 files.
            Each HDF5 file should contain a 'main' dataset representing the neuron volume.
        vesicle_h5_path (str): Path to the HDF5 file containing vesicle data.
            Each dataset within this file should correspond to a neuron's name and
            contain the vesicle volume data.
        offset_csv_path (str): Path to the CSV file containing offset information for each neuron.
            The CSV should have a column for neuron names and corresponding x, y, z offsets.
        voxel_resolution (tuple, optional): Voxel resolution in (z, y, x) order.
            Defaults to (30, 64, 64).
        bind_address (str, optional): IP address to bind the server to. Defaults to 'localhost'.
        verbose (bool, optional): Enable verbose logging. Defaults to True.
    Returns:
        neuroglancer.Viewer: The Neuroglancer viewer object if successful, None otherwise.
            The viewer will display the neuron and vesicle data as segmentation layers,
            with appropriate offsets applied.
    Raises:
        FileNotFoundError: If the vesicle HDF5 file is not found.
        KeyError: If a neuron name in the neuron directory or offset CSV is not found
            as a dataset in the vesicle HDF5 file.
    Notes:
        - The function loads neuron and vesicle data, creates segmentation layers in Neuroglancer,
            and applies offsets to align the data.
        - It iterates through the HDF5 files in the specified neuron directory.
        - It expects the vesicle HDF5 file to contain datasets named after the neurons.
        - It uses a CSV file to determine the spatial offset for each neuron.
        - The function attempts to handle missing data gracefully by skipping neurons
            with missing offset or vesicle data.
        - It explicitly deletes neuron and vesicle data and calls garbage collection
            to manage memory usage.
    """

    port = find_free_port()
    neuroglancer.set_server_bind_address(bind_address=bind_address, bind_port=port)
    viewer = neuroglancer.Viewer()

    if verbose:
        print(f"Starting Neuroglancer server on {bind_address}:{port}")

    offsets = load_offsets_csv(offset_csv_path)
    if offsets is None:
        logging.error("Failed to load offsets, exiting")
        return None

    if verbose:
        print(f"Loaded offsets for {len(offsets)} neurons")

    try:
        with h5py.File(vesicle_h5_path, 'r') as vesicle_h5:
            with viewer.txn() as s:
                neuron_files = [f for f in os.listdir(neuron_h5_dir) if f.endswith('.h5')]
                if verbose:
                    print(f"Processing {len(neuron_files)} neuron files...")
                
                for fname in neuron_files:
                    name = os.path.splitext(fname)[0]
                    if name not in offsets or name not in vesicle_h5:
                        if verbose:
                            logging.warning(f"Skipping {name}: missing offset or vesicle data")
                        continue

                    offset = offsets[name]
                    neuron_path = os.path.join(neuron_h5_dir, fname)
                    neuron_data = load_h5_volume(neuron_path, "main")
                    if neuron_data is None:
                        logging.error(f"Failed to load neuron data for {name}, skipping")
                        continue
                    
                    try:
                        vesicle_data = np.array(vesicle_h5[name])
                    except KeyError:
                        logging.error(f"Failed to load vesicle data for {name}, skipping")
                        continue

                    append_seg_layer(s, f'neuron_{name}', neuron_data, voxel_resolution, offset)
                    append_seg_layer(s, f'vesicles_{name}', vesicle_data, voxel_resolution, offset)

                    if verbose:
                        logging.info(f"Loaded {name} neuron & vesicles into viewer")
                    del neuron_data, vesicle_data
                    gc.collect()

    except FileNotFoundError:
        logging.error(f"Vesicle HDF5 file not found: {vesicle_h5_path}")
        return None

    print("Neuroglancer viewer is live:")
    print(viewer)
    return viewer


def main():
    """Main function to handle command-line arguments and launch Neuroglancer viewer."""
    parser = argparse.ArgumentParser(
        description="Launch Neuroglancer viewer for neuron and vesicle visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "neuron_h5_dir",
        type=str,
        help="Directory containing neuron HDF5 files (each with 'main' dataset)"
    )
    
    parser.add_argument(
        "vesicle_h5_path",
        type=str,
        help="Path to HDF5 file containing vesicle data (datasets named by neuron)"
    )
    
    parser.add_argument(
        "offset_csv",
        type=str,
        help="Path to CSV file with neuron offsets (columns: name, z, y, x)"
    )
    
    parser.add_argument(
        "-r", "--resolution",
        type=float,
        nargs=3,
        default=[30.0, 64.0, 64.0],
        metavar=("Z", "Y", "X"),
        help="Voxel resolution in nanometers (Z Y X)"
    )
    
    parser.add_argument(
        "-a", "--address",
        type=str,
        default="localhost",
        help="IP address to bind Neuroglancer server to"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the viewer open (blocks until manually stopped)"
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    if not Path(args.neuron_h5_dir).is_dir():
        print(f"Error: Neuron directory not found: {args.neuron_h5_dir}")
        return 1
    
    if not Path(args.vesicle_h5_path).is_file():
        print(f"Error: Vesicle HDF5 file not found: {args.vesicle_h5_path}")
        return 1
    
    if not Path(args.offset_csv).is_file():
        print(f"Error: Offset CSV file not found: {args.offset_csv}")
        return 1
    
    try:
        viewer = launch_neuroglancer(
            neuron_h5_dir=args.neuron_h5_dir,
            vesicle_h5_path=args.vesicle_h5_path,
            offset_csv_path=args.offset_csv,
            voxel_resolution=tuple(args.resolution),
            bind_address=args.address,
            verbose=not args.quiet
        )
        
        if viewer is None:
            print("Failed to launch Neuroglancer viewer")
            return 1
        
        if not args.quiet:
            print(f"\nNeuroglancer viewer successfully launched!")
            print(f"   Open the URL above in your browser to view the data.")
        
        if args.keep_open:
            if not args.quiet:
                print("\nKeeping viewer open. Press Ctrl+C to stop...")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                if not args.quiet:
                    print("\nShutting down Neuroglancer viewer...")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
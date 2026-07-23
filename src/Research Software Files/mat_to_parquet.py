"""
This first file takes .mat files and converts them to .parquet files.
.parquet files are able to handle large files efficiently

This uses a CLI approach so that the data is never integral to the file and can be used across devices

"""

import argparse
from pathlib import Path

import h5py
import polars as pl
import numpy as np

import sys

def read_mat_file(filepath: Path) -> dict: # Reads in the .mat file 
    parsed_data = {}
    with h5py.File(filepath, "r") as h5_file:
        for key in h5_file.keys():
            # Skip MATLAB internal metadata groups
            if key.startswith("#"):
                continue
                
            obj = h5_file[key]
            
            # Ensure we are reading a Dataset, not an HDF5 Sub-group
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                
                # Convert to numpy array and flatten
                data_array = np.asarray(data).squeeze()
                
                # Squeeze scalar 0D values to 1D so Polars can frame them
                if data_array.ndim == 0:
                    data_array = data_array.reshape(1)
                else:
                    data_array = data_array.flatten()
                    
                parsed_data[key] = data_array

    # Equalize array lengths so Polars doesn't crash on length mismatches
    if parsed_data:
        max_len = max(len(arr) for arr in parsed_data.values())
        for key, arr in parsed_data.items():
            if len(arr) < max_len:
                # Pad shorter arrays with NaN
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                parsed_data[key] = padded

    return parsed_data
def mat_to_parquet(input_path: Path, output_dir: Path) -> Path: #converts a .mat file to a compressed .parquet
    
    #ensure input esists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    #ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #Load data from .mat file
    data_dict = read_mat_file(input_path)
    
    #convert to polars dataframe
    df = pl.DataFrame(data_dict)
    
    #Output file name
    output_file = output_dir / f"{input_path.stem}.parquet"
    
    #write to parquet ("snappy" = low disk usage and fast reading/writing)
    df.write_parquet(output_file, compression ="snappy")
    
    return output_file

def main(): #CLI configuration
    parser = argparse.ArgumentParser(
        description="Convert .mat sensor files to a .parquet format")
    
    parser.add_argument( #adds input
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to a single .mat file OR a directory with multiplie .mat files", 
        )
    
    parser.add_argument( #adds output
        "-o",
        "--output",
        type=str,
        default="./data/processed",
        help="Directory where output .parquet files should be saved (default: ./data/processed)"
        )
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Gather target .mat files
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = list(input_path.glob("*.mat"))
        if not files_to_process:
            print(f"No .mat files found in directory: {input_path}")
            sys.exit(1)
    else:
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Found {len(files_to_process)} file(s) to process.\n")

    # Process files sequentially
    for file in files_to_process:
        print(f"Processing: {file.name} ...", end="", flush=True)
        try:
            out_file = mat_to_parquet(file, output_dir)
            print(f" Done! Saved to -> {out_file}")
        except Exception as e:
            print(f" Failed! Error: {e}")


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
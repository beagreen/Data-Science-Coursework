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

def read_mat_file(filepath: Path) -> dict:
    parsed_data = {}

    with h5py.File(filepath, "r") as h5_file:
        # find the top-level struct name (ignoring #refs# and metadata)
        top_keys = [k for k in h5_file.keys() if not k.startswith("#")]
        if not top_keys:
            return {}

        struct_group = h5_file[top_keys[0]]

        # iterate through all field names in the struct 
        for field_name in struct_group.keys():
            if field_name.startswith("#"):
                continue

            field_obj = struct_group[field_name]

            # Direct numeric dataset
            if isinstance(field_obj, h5py.Dataset):
                data = field_obj[()]
                if data.dtype == "object":  # Contains references to array objects
                    extracted_runs = []
                    for ref_cell in data.flat:
                        try:
                            # Dereference HDF5 object pointer
                            ref_data = h5_file[ref_cell][()].squeeze()
                            extracted_runs.append(ref_data)
                        except Exception:
                            continue

                    # Concatenate or flatten signal across runs
                    if extracted_runs:
                        parsed_data[field_name] = np.concatenate([r.flatten() for r in extracted_runs if r.size > 0])
                else:
                    parsed_data[field_name] = data.squeeze().flatten()

    # Equalize array lengths with NaNs so Polars creates a valid DataFrame
    if parsed_data:
        max_len = max(len(arr) for arr in parsed_data.values())
        for key, arr in parsed_data.items():
            if len(arr) < max_len:
                padded = np.full(max_len, np.nan)
                padded[: len(arr)] = arr
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
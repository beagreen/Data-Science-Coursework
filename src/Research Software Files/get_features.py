# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:12:29 2026

@author: map25bg

This file will be used to extract the wawnted features from the created .parquet files. 
The targeted features are statistical  processing features

For the intended purpose of this script use it after mat_to_parquet to get statistical features from the .parquet files
It will create extracted_features.csv and extracted_features.parquet
This file will calculate statistical features from segments of the data, this ensures that the dataset is large enough for training a ML model
Otherwise, taking features once from all of the data reduces the dataset too much

Statistics help:
    crest factor: analysis of the spikiness of a waveform (Peak Amp/RMS)
    skewness: measures the asymmetry of a probability distribution about its mean
    kurtosis: statistical measure that desicribes how extreme the edges of a probability distribution. It is a measure of how often outliers occue

Uses a command line interface as laid out in the README
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.stats import kurtosis, skew
from tqdm import tqdm

#calculating time domain features from a signal segment
def feature_calculate(signal: np.ndarray) -> dict: 
    
    #handle nans
    clean_signal = signal[~np.isnan(signal)]
    
    if len(clean_signal) == 0:
        return {}
    
    #time domain features
    mean_value = float(np.mean(clean_signal))
    std_value = float(np.std(clean_signal))
    rms_value = float(np.sqrt(np.mean(clean_signal**2)))
    peak_value = float(np.max(np.abs(clean_signal)))
    crest_factor = peak_value / rms_value if rms_value != 0 else 0.0
    
    #if signal constant (standard near 0) then automatically set skewness and kurtosis to 0.0 - this prevents 'catestrophic cancellations) and sets to 0 not NaN
    
    if std_value < 1e-8:
        skewness_val = 0.0
        kurtosis_val = 0.0
    else:
        skewness_val = float(skew(clean_signal))
        kurtosis_val = float(kurtosis(clean_signal))
        
    features = {
        "mean"  : mean_value,
        "standard_deviation": std_value,
        "variance": float(np.var(clean_signal)),
        "rms": rms_value,
        "peak_to_peak": float(np.ptp(clean_signal)),
        "crest_factor": crest_factor,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val
        }
    return features

# extract the time domain features for each channel of the parquet file, segments each signal into window_size sections and then calculates
def parquet_extract(filepath: Path, window_size: int = 5000) -> list[dict]:
    df = pl.read_parquet(filepath)
    file_features = []

    num_rows = len(df)
    num_windows = num_rows // window_size
    
    #backstop if file too small
    if num_windows == 0:
        num_windows = 1
        window_size = num_rows
        
    # iterate through windows
    for win_idx in range(num_windows):
        start_idx = win_idx * window_size
        end_idx = start_idx + window_size
        
        
        for col_name in df.columns:
            signal_window = df[col_name][start_idx:end_idx].to_numpy()
            feats = feature_calculate(signal_window)
    
            if feats:
                feats["file_name"] = filepath.name
                feats["window_id"] = win_idx
                feats["channel"] = col_name
                file_features.append(feats)

    return file_features

#parser to allow for CLI
def main():
    parser = argparse.ArgumentParser(description="Extract time-domain features from .parquet files.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input .parquet file or directory")
    parser.add_argument("-o", "--output", type=str, default="./data/features", help="Output directory")
    parser.add_argument("-w", "--window-size", type=int, default=5000, help="Number of samples per window (default: 5000)")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather target files
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.parquet"))
    else:
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Extracting time-domain features from {len(files)} file(s)...")

    all_results = []
    for f in tqdm(files, desc="Processing Files", unit="file"):
        results = parquet_extract(f, window_size=args.window_size)
        all_results.extend(results)
        
    if not all_results:
        print("No valid data processed.")
        return

    # Export to Polars DataFrame
    summary_df = pl.DataFrame(all_results)

    csv_out = output_dir / "extracted_features.csv"
    parquet_out = output_dir / "extracted_features.parquet"
    
    try:
        summary_df.write_csv(csv_out)
        summary_df.write_parquet(parquet_out)
    
        print(f"\n Done! Extracted features across {len(summary_df)} channels.")
        print(f"Saved outputs to:\n  -> {csv_out}\n  -> {parquet_out}")
    except OSError:
        print("\n Permission Error: Could not save features!")
        print("Please CLOSE 'extracted_features.csv' if you have it open in Excel and try again.")
    


if __name__ == "__main__":
    main()
























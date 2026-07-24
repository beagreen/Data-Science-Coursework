# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:12:29 2026

@author: map25bg

This file will be used to extract the wawnted features from the created .parquet files. 
The targetted features are statistical  processing features

Statistics help:
    crest factor: analysis of the spikiness of a waveform (Peak Amp/RMS)
    skewness: measures the asymmetry of a probability distribution about its mean
    kurtosis: statistical measure that desicribes how extreme the edges of a probability distribution. It is a measure of how often outliers occue

"""

import argparse
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.stats import kurtosis, skew

#calculating time and freq. domain features from the signal
def feature_calculate(signal: np.ndarray, sample_rate: float = 1000.0) -> dict: 
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
    
    features = {
        "mean"  : mean_value,
        "standard_deviation": std_value,
        "variance": float(np.var(clean_signal)),
        "rms": rms_value,
        "peak_to_peak": float(np.ptp(clean_signal)),
        "crest_factor": crest_factor,
        "skewness": float(skew(clean_signal)),
        "kurtosis": float(kurtosis(clean_signal))
        }
    return features

# extract the time domain features for each channel of the parquet file
def parquet_extract(filepath: Path) -> list[dict]:
    df = pl.read_parquet(filepath)
    file_features = []

    for col_name in df.columns:
        signal = df[col_name].to_numpy()
        feats = feature_calculate(signal)

        if feats:
            feats["file_name"] = filepath.name
            feats["channel"] = col_name
            file_features.append(feats)

    return file_features

#parser to allow for CLI
def main():
    parser = argparse.ArgumentParser(description="Extract time-domain features from .parquet files.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input .parquet file or directory")
    parser.add_argument("-o", "--output", type=str, default="./data/features", help="Output directory")

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
    for f in files:
        results = parquet_extract(f)
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
























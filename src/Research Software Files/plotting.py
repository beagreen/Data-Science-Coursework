# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:13:55 2026

@author: map25bg

This file is designed to plot and visualise the features classified in classify.py
It has been designed specifically for the data used in this project. It will assign labels based on those expected from the given Data

Uses a command line interface as laid out in the README
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler


#assigns class names to the data groups    
def assign_classes(target_str: str) -> str:
    target_lower = str(target_str).lower()
    
    if "linear" in target_lower:
        return "Linear"
    elif "machining" in target_lower:
        return "Machining"
    elif "5000" in target_lower:
        return "Spindle 5000rpm"
    elif "12000" in target_lower:
        return "Spindle 12000rpm"
    else:
        return "Other Regime"

#generates a confusion matrix for each operational class
def plot_confusion_matrices(results_path: Path, output_dir: Path):
    if not results_path.exists():
        print(f"Error: Output file not found at '{results_path}'")
        return
    
    df = pd.read_csv(results_path)
    # Assign each row to its operational category
    df["regime"] = df["y_true"].apply(assign_classes)

    unique_regimes = sorted(df["regime"].unique())
    print(f"Identified operational regimes: {unique_regimes}")

    for regime in unique_regimes:
        subset = df[df["regime"] == regime]
        
        def strip_labels(label_str: str) -> str:
            cleaned = str(label_str).replace("Segmented ", "")
            prefixes_to_remove = [
                "Spindle5000 ",
                "Spindle12000 ",
                "Linear ",
                "Machining ",
                ]
            for prefix in prefixes_to_remove:
                cleaned = cleaned.replace(prefix, ""). strip()
            return cleaned
        
        subset["y_true_clean"] = subset["y_true"].apply(strip_labels)
        subset["y_pred_clean"] = subset["y_pred"].apply(strip_labels)

        # Get unique labels present within this operational regime
        labels = sorted(subset["y_true_clean"].unique())
        

        if len(labels) <= 1:
            print(
                f"Skipping plot for '{regime}' (contains only 1 target class).")
            continue

        cm = confusion_matrix(
            subset["y_true_clean"], subset["y_pred_clean"], labels=labels, normalize="true"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels
        )
        disp.plot(cmap="Blues", ax=ax, values_format=".2f")
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        
        ax.set_xlabel("Predicted label", labelpad=10, fontsize=14, fontweight="bold")
        ax.set_ylabel("True label", labelpad=10, fontsize=14, fontweight="bold")

        ax.set_title(
            f"Classification Performance: {regime}\n(Normalized Confusion Matrix)",
            pad=15, fontsize=16, fontweight="bold",
        )
        ax.grid(False)

        # Save plot for this operational class
        safe_regime_name = regime.replace(" ", "_").replace("/", "_").lower()
        save_path = output_dir / f"confusion_{safe_regime_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
        

def main():
    parser = argparse.ArgumentParser(
        description="Generate all confusion matrices and cluster scatter plots."
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        required=True,
        help="Path to extracted_features.parquet or CSV",
    )
    parser.add_argument(
        "-r",
        "--results",
        type=str,
        required=True,
        help="Path to classification_results.csv (from classify.py)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data_results",
        help="Directory to save generated plots",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    

    print("=========================================================")
    print("STARTING COMPLETE VISUALIZATION SUITE")
    print("=========================================================")

    plot_confusion_matrices(Path(args.results), output_dir)
    
    print("\n All figures saved successfully to:", output_dir)


if __name__ == "__main__":
    main()
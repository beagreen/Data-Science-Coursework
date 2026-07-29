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
        
def plot_3d_scatter(df: pd.DataFrame, target_classes: list, x_col: str, y_col: str, z_col: str, title: str, save_path: Path):
    
    subset = df[df["target"].isin(target_classes)].copy()
    if len(subset) == 0:
        
        print(f"Skipping '{title}': Target classes not found.")
        return
    
    for col in [x_col, y_col, z_col]:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset.dropna(subset=[x_col, y_col, z_col])

    if len(subset) == 0:
        print(f"Skipping '{title}': Missing required columns {x_col}, {y_col}, {z_col}.")
        return

    # Standardize features (Z-score scaling) so axes range from -3 to +3
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(subset[[x_col, y_col, z_col]])

    subset["X_scaled"] = scaled_vals[:, 0]
    subset["Y_scaled"] = scaled_vals[:, 1]
    subset["Z_scaled"] = scaled_vals[:, 2]

    # Set up 3D plot
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    colours = ["red", "green", "blue", "yellow"]
    markers = ["+", "o", "^", "x", "s"]
    
    unique_targets = subset["target"].unique()

    for i, target in enumerate(unique_targets):
        t_data = subset[subset["target"] == target]

        colour = colours[i % len(colours)]
        marker = markers[i % len(markers)]

        ax.scatter(
            t_data["X_scaled"],
            t_data["Y_scaled"],
            t_data["Z_scaled"],
            label=target,
            c=colour,
            marker=marker,
            alpha=0.6,
            s=25,
            edgecolors="none" if marker != "o" else "face",
        )

    # Set labels based on the chosen features
    ax.set_xlabel(x_col.replace("_", " "), labelpad=10)
    ax.set_ylabel(y_col.replace("_", " "), labelpad=10)
    ax.set_zlabel(z_col.replace("_", " "), labelpad=2)
    

    ax.set_title(title, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True)

    # Enable grid lines
    ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust view angle 
    ax.view_init(elev=20, azim=-60)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Saved 3D plot: {save_path.name}")
    
def generate_all_clusters(features_path: Path, output_dir: Path):
    if not features_path.exists():
        print(f"Skipping Cluster Plots: Could not find '{features_path}'")
        return

    print("\n--- Generating 3D Feature Scatter Plots ---")
    df = (
        pd.read_parquet(features_path)
        if features_path.suffix == ".parquet"
        else pd.read_csv(features_path)
    )

    if "channel" in df.columns:
        df = df.pivot(
            index=["file_name", "window_id"],
            columns="channel",
            values=[
                "mean",
                "standard_deviation",
                "variance",
                "rms",
                "peak_to_peak",
                "crest_factor",
                "skewness",
                "kurtosis",
            ],
        )
        # Format columns as 'Channel_Metric' e.g. 'SpindleAccX_peak_to_peak'
        df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
        df = df.reset_index()

    def get_target(file_name):
        stem = Path(str(file_name)).stem.replace("-", "_")
        parts = [
            p
            for p in stem.split("_")
            if not p.lower().startswith("run")
            and not p.isdigit()
            and p.lower() != "data"
        ]
        return " ".join(parts)

    df["target"] = df["file_name"].apply(get_target)
    unique_targets = sorted(df["target"].unique())

    # --- DEFINE 3D TARGET COLUMNS ---
    # Automatically match columns or use default dynamic acceleration columns
    all_cols = df.columns.tolist()

    x_col = next(
        (c for c in all_cols if "accx" in c.lower() and "peak" in c.lower()),
        next((c for c in all_cols if "peak" in c.lower()), None),
    )
    y_col = next(
        (c for c in all_cols if "accy" in c.lower() and "kurtosis" in c.lower()),
        next((c for c in all_cols if "kurtosis" in c.lower()), None),
    )
    z_col = next(
        (c for c in all_cols if "accx" in c.lower() and "mean" in c.lower()),
        next((c for c in all_cols if "mean" in c.lower()), None),
    )

    print(f"3D Axes Columns -> X: {x_col} | Y: {y_col} | Z: {z_col}")

    # Generate 3D Plots
    spindle5000 = [t for t in unique_targets if "5000" in t.lower()]
    if spindle5000:
        plot_3d_scatter(
            df,
            spindle5000,
            x_col,
            y_col,
            z_col,
            "Cluster 1: Spindle 5000 RPM Regimes",
            output_dir / "cluster_spindle5000.png",
        )

    spindle12000 = [t for t in unique_targets if "12000" in t.lower()]
    if spindle12000:
        plot_3d_scatter(
            df,
            spindle12000,
            x_col,
            y_col,
            z_col,
            "Cluster 2: Spindle 12000 RPM Regimes",
            output_dir / "cluster_spindle12000.png",
        )

    machining = [t for t in unique_targets if "machining" in t.lower()]
    if machining:
        plot_3d_scatter(
            df,
            machining,
            x_col,
            y_col,
            z_col,
            "Cluster 3: Machining Fault Regimes",
            output_dir / "cluster_machining.png",
        )

    linear = [t for t in unique_targets if "linear" in t.lower()]
    if linear:
        plot_3d_scatter(
            df,
            linear,
            x_col,
            y_col,
            z_col,
            "Cluster 4: Linear Axis Motion Regimes",
            output_dir / "cluster_linear.png",
        )

# =============================================================================
# MAIN CLI ENTRYPOINT
# =============================================================================

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
    generate_all_clusters(Path(args.features), output_dir)

    print("\n All figures saved successfully to:", output_dir)


if __name__ == "__main__":
    main()
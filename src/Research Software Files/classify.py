"""
Created on Fri Jul 24 13:23:04 2026

@author: map25bg

This script is written to classify the extracted features. 
First by splitting the features from the target labels
Then 80% of the data will be used to test the model, using the remaining 20% for testing of a Random Forest ML model

It will take features from a .parquet or a .csv file type
"""
import argparse
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#read file into a pandas dataframe
def load_prep_data(feature_path: Path):
    if feature_path.suffix == ".parquet":
        df = pd.read_parquet(feature_path)
    else:
        df = pd.read_csv(feature_path)
    
    #pivot the table so that random forest can see all channels at once 
    df_pivoted = df.pivot(
        index=["file_name", "window_id"],
        columns="channel", 
        values=["mean", 
                "standard_deviation", 
                "variance", 
                "rms", 
                "peak_to_peak", 
                "crest_factor", 
                "skewness", 
                "kurtosis"]
    )
    
    # Flatten column names (e.g., ('rms', 'SpindleAccZ') -> 'SpindleAccZ_rms')
    df_pivoted.columns = [f"{col[1]}_{col[0]}" for col in df_pivoted.columns]
    df_pivoted = df_pivoted.reset_index()
    
    
# extract the right category from the filename - designed for the target data but will work for other filenames too
    def categorise_from_filename(name: str) -> str:
        name_str = str(name)
        if "Baseline" in name_str:
            return "Baseline"
        elif "Heavy" in name_str:
            return "Heavy Tool"
        elif "Override" in name_str:
            return "Feedrate Adjusted"
        elif "Misalignment" in name_str:
            return "Misalignment"
        elif "SurfaceCracks" in name_str:
            return "Surface Cracks"
        elif "ToolWear" in name_str:
            return "Tool Wear"
        elif "Unbalanced" in name_str:
            return "Unbalanced"
        else: #Using filename as the label
            return Path(name_str).stem.split("_")[0]
    
    #create both feature and target dataframes (features = the statistical data from the experimental trial, while targets = the experimental categories)
    df_pivoted["target"] = df_pivoted["file_name"].apply(categorise_from_filename)

    return df_pivoted

def chronological_split(df_pivoted: pd.DataFrame, train_ratio: float = 0.8): #80/20 chronological split per file
    train_indices = []
    test_indices = []
    
    #sort into sequential window order
    df_pivoted = df_pivoted.sort_values(
        by=["file_name", "window_id"]
    ).reset_index(drop=True)
    
    # Split each individual file by time sequence
    for file_name, group in df_pivoted.groupby("file_name"):
        num_windows = len(group)
        split_point = int(num_windows * train_ratio)

        # First 80% of window_ids -> Train
        train_indices.extend(group.index[:split_point])
        # Last 20% of window_ids -> Test
        test_indices.extend(group.index[split_point:])

    ignore_cols = ["file_name", "window_id", "target"]
    feature_cols = [c for c in df_pivoted.columns if c not in ignore_cols]

    X_train = df_pivoted.iloc[train_indices][feature_cols]
    y_train = df_pivoted.iloc[train_indices]["target"]

    X_test = df_pivoted.iloc[test_indices][feature_cols]
    y_test = df_pivoted.iloc[test_indices]["target"]

    return X_train, X_test, y_train, y_test, feature_cols
    

def run_classification(df_pivoted: pd.DataFrame, train_ratio: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_cols = chronological_split(
        df_pivoted, train_ratio=train_ratio
    )

    print(
        f"Performing {int(train_ratio*100)}/{int((1-train_ratio)*100)} chronological per-file split..."
    )
    print(f"Total Train Samples (first 80% time): {len(X_train)}")
    print(f"Total Test Samples (last 20% time):  {len(X_test)}")
    print(
        f"Unique Classes in Train: {y_train.nunique()} | Unique Classes in Test: {y_test.nunique()}"
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=24)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
        
    
    # printing a 'classification report' to visually see the outcomes of the random forest analysis
    print("\n" + "=" * 65)
    print("MULTI-CLASS CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    #save outputs for plotting
    results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    results_path = output_dir / "classification_results.csv"
    results_df.to_csv(results_path, index=False)
    
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_}).sort_values(by="importance", ascending=False)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    print(f"\n Data exported for plotting:\n  -> {results_path}\n  -> {importance_path}")
    
    
def main():
    parser = argparse.ArgumentParser(description="Multi-class fault classification.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to extracted_features.parquet/csv")
    parser.add_argument("-o", "--output", type=str, default="./data/results", help="Directory to save plots")
    parser.add_argument("-r", "--ratio", type=float, default=0.80, help="Training data ratio (default: 0.80 for 80/20 split)",)

    args = parser.parse_args()
    df_pivoted = load_prep_data(Path(args.input))
    run_classification(df_pivoted, args.ratio, Path(args.output))

    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
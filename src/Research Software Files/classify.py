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
from sklearn.model_selection import train_test_split


#read file into a pandas dataframe
def load_prep_data(feature_path: Path):
    if feature_path.suffix == ".parquet":
        df = pd.read_parquet(feature_path)
    else:
        df = pd.read_csv(feature_path)
    
    #pivot the table so that random forest can see all channels at once 
    df_pivoted = df.pivot(
        index="file_name", 
        columns="channel", 
        values=["mean", "standard_deviation", "variance", "rms", "peak_to_peak", "crest_factor", "skewness", "kurtosis"]
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
    
    #creaate both feature and target dataframes (features = the statistical data from the experimental trial, while targets = the experimental categories)
    df_pivoted["target"] = df_pivoted["file_name"].apply(categorise_from_filename)

    ignore_cols = ["file_name", "target"]
    feature_cols = [c for c in df_pivoted.columns if c not in ignore_cols]

    X_features = df_pivoted[feature_cols]
    y_targets = df_pivoted["target"]

    return X_features, y_targets, feature_cols

def run_classification(X_features, y_targets, feature_names, output_dir: Path):
    output_dir.mkdir(parents = True, exist_ok = True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_targets, 
                                                        test_size=0.2, random_state=41, stratify=y_targets)

#creating the random forest decision trees
    print(f"Training multi-class Random Forest on {len(y_targets.unique())} target classes...")
    clf = RandomForestClassifier(n_estimators=100, random_state=24)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    # printing a 'classification report' to visually see the outcomes of the random forest analysis
    print("\n" + "=" * 65)
    print("MULTI-CLASS CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_test, y_pred))
    
def main():
    parser = argparse.ArgumentParser(description="Multi-class fault classification.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to extracted_features.parquet/csv")
    parser.add_argument("-o", "--output", type=str, default="./data/results", help="Directory to save plots")

    args = parser.parse_args()
    X_features, y_targets, feature_names = load_prep_data(Path(args.input))
    run_classification(X_features, y_targets, feature_names, Path(args.output))
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
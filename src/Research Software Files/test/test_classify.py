# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:15:34 2026

@author: map25bg

test file fpr classify.py

"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify import (load_prep_data, chronological_split, run_classification, main)

class TestClassify(unittest.TestCase):
    
    def setUp(self):
        #temporary folder created before each test
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        
    def tearDown(self):
        #deletes the temporary folder
        self.temp_dir_obj.cleanup()
        
    def create_features_parquet(self, filename="features.parquet") -> Path:
        data=[]
        files = ["Linear_Baseline_run1.csv", "Linear_Heavy_run1.csv"]
        channels = ["Channel1", "Channel2"]
        
        #create 10 sequential windows
        for file_name in files:
            for win_id in range(10):
                for ch in channels:
                    data.append({"file_name": file_name,
                            "window_id": win_id,
                            "channel": ch,
                            "mean": np.random.rand(),
                            "standard_deviation": np.random.rand(),
                            "variance": np.random.rand(),
                            "rms": np.random.rand(),
                            "peak_to_peak": np.random.rand(),
                            "crest_factor": np.random.rand(),
                            "skewness": np.random.rand(),
                            "kurtosis": np.random.rand()})
                    
        df = pd.DataFrame(data)
        file_path = self.temp_dir / filename
        df.to_parquet(file_path, index=False)
        return file_path
    
    def test_load_prep_data(self):
        parquet_path = self.create_features_parquet("load_test.parquet")
        df_pivoted = load_prep_data(parquet_path)
        
        #assert necessary columns exist
        self.assertIn("file_name", df_pivoted.columns)
        self.assertIn("window_id", df_pivoted.columns)
        self.assertIn("target", df_pivoted.columns)
        
        #assert un-pivoted feature columns are created cleanly
        self.assertIn("Channel1_mean", df_pivoted.columns)
        self.assertIn("Channel2_rms", df_pivoted.columns)
        
        #assert that run1 was stripped and extract the target label
        unique_targets = set(df_pivoted["target"].unique())
        self.assertEqual(unique_targets, {"Linear Baseline", "Linear Heavy"})
        
    def test_chronological_split(self): #test 80/20 chronological split
        parquet_path = self.create_features_parquet("split_test.parquet")
        df_pivoted = load_prep_data(parquet_path)
        
        X_train, X_test, y_train, y_test, feature_cols = chronological_split(df_pivoted, train_ratio=0.80)
        
        self.assertEqual(len(X_train), 16) #10 windows per file, 2 files = 80% of 20 = 16
        self.assertEqual(len(X_test), 4)
        
        self.assertEqual(len(y_train), 16)
        self.assertEqual(len(y_test), 4)
        
        self.assertNotIn("target", feature_cols)
        self.assertNotIn("file_name", feature_cols)
        self.assertNotIn("window_id", feature_cols)
        
    def test_run_classification(self):
        parquet_path = self.create_features_parquet("class_test.parquet")
        df_pivoted = load_prep_data(parquet_path)
        output_dir = self.temp_dir / "classification_results"
        
        run_classification(df_pivoted, 0.8, output_dir)
        
        results_path = output_dir / "classification_results.csv"
        importance_path = output_dir / "feature_importance.csv"
        
        self.assertTrue(results_path.exists(), "classification_results.csv was not created")
        self.assertTrue(importance_path.exists(), "feature_importance.csv was not created")
        
        results_df = pd.read_csv(results_path)
        self.assertIn("y_true", results_df.columns)
        self.assertIn("y_pred", results_df.columns)
        self.assertEqual(len(results_df), 4)
        
    def test_main(self):
        parquet_path = self.create_features_parquet("cli_test.parquet")
        output_dir = self.temp_dir / "cli_results"
        
        test_args = [
            "classify.py",
            "-i", str(parquet_path),
            "-o", str(output_dir),
            "-r", "0.80",
        ]
        
        #temporarily mock sys.argv so argparse reads the above test args
        original_argv = sys.argv
        try:
            sys.argv = test_args
            main()
        finally:
            sys.argv = original_argv #restore sys.argv
            
        expected_results = output_dir / "classification_results.csv"
        self.assertTrue(expected_results.exists(), "CLI main failed to generate output")
        self.assertGreater(expected_results.stat().st_size, 0, "Generated results CSV is empty")
        
if __name__ == "__main__":
    unittest.main()
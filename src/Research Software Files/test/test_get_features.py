# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:34:53 2026

@author: map25bg

This file is built to test get_features.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from get_features import feature_calculate, main, parquet_extract

class TestGetFeatures(unittest.TestCase):
    
    def setUp(self):
        #temporary folder created before each test
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        
    def tearDown(self):
        #deletes the temporary folder
        self.temp_dir_obj.cleanup()
        
    def temporary_parquet_file(self, filename="sample.parquet", samples=10000) -> Path:
        parquet_path = self.temp_dir / filename
        data = {"AccX": np.linspace(-1.0, 1.0, samples), "AccY": np.sin(np.linspace(0, 10, samples))}
        df = pl.DataFrame(data)
        df.write_parquet(parquet_path)
        return parquet_path
    
    def test_feature_calculate(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = feature_calculate(signal)
        
        #dictionary assertions
        self.assertIsInstance(features, dict)
        self.assertEqual((features["mean"]), 3.0)
        self.assertEqual(features["peak_to_peak"], 4.0)
        self.assertIn("crest_factor", features)
        self.assertIn("skewness", features)
        
        # check back stop for flat signals 
        flat_signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])        
        flat_features = feature_calculate(flat_signal)
        self.assertEqual(flat_features["skewness"], 0.0)
        self.assertEqual(flat_features["kurtosis"], 0.0)
        
    def test_parquet_extract(self):
        parquet_path = self.temporary_parquet_file("test_extract.parquet")
        feature_list = parquet_extract(parquet_path, window_size=5000)
        
        self.assertIsInstance(feature_list, list)
        self.assertEqual(len(feature_list), 4) #10000 sample file and 5000 window size: 2 windows x 2 channels = 4 feature rows
        
        # check keys added during parquet extraction
        first_row = feature_list[0]
        self.assertEqual(first_row["file_name"], "test_extract.parquet")
        self.assertIn("window_id", first_row)
        self.assertIn("channel", first_row)
        
    def test_main(self):
        parquet_path = self.temporary_parquet_file("cli_sample.parquet", samples=10000)
        output_dir = self.temp_dir / "features_output"

        test_args = [
            "get_features.py",
            "-i", str(parquet_path),
            "-o", str(output_dir),
            "-w", "5000",
        ]

        #temporarily mock sys.argv so argparse reads the above test args
        original_argv = sys.argv
        try:
            sys.argv = test_args
            main()
        finally:
            sys.argv = original_argv #restore sys.argv
            
        expected_csv = output_dir / "extracted_features.csv"
        expected_parquet = output_dir / "extracted_features.parquet"
        
        self.assertTrue(expected_csv.exists(), "CLI failed to create extracted_features.csv")
        self.assertTrue(expected_parquet.exists(), "CLI failed to create extracted_features.parquet")
        self.assertGreater(expected_csv.stat().st_size, 0, "Generated CSV file is empty!")
        
if __name__ == "__main__":
    unittest.main()
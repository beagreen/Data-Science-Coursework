# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:06:47 2026

@author: map25bg

test file for plotting.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting import assign_classes, plot_confusion_matrices, main

class TestPlotting(unittest.TestCase):
    
    def setUp(self):
        #temporary folder created before each test
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        
    def tearDown(self):
        #deletes the temporary folder
        self.temp_dir_obj.cleanup()
        
    def create_temp_results_csv(self, filename="results.csv") -> Path:
        data = {
            "y_true": [
                "Segmented Linear Baseline",
                "Segmented Linear Heavy",
                "Segmented Linear Override",
                "Spindle5000 Baseline",
                "Spindle5000 Heavy",
                "Spindle5000 Override"],
            "y_pred": [
                "Segmented Linear Baseline",
                "Segmented Linear Heavy",
                "Segmented Linear Override",
                "Spindle5000 Baseline",
                "Spindle5000 Unbalanced", #misclassification
                "Spindle5000 Override"]}
        df = pd.DataFrame(data)
        file_path = self.temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_assign_classes(self):
        test_cases = [
            ("Segmented_Linear_Baseline", "Linear"),
            ("Machining_ToolWear_Run1", "Machining"),
            ("Spindle_5000rpm_Heavy", "Spindle 5000rpm"),
            ("Spindle_12000rpm_Override", "Spindle 12000rpm"),
            ("Random_Unknown_Regime", "Other Regime"),
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                self.assertEqual(assign_classes(input_str), expected)
                
    def test_plot_confusion_matrices(self):
        results_path = self.create_temp_results_csv("test_results.csv")
        output_dir = self.temp_dir / "plots"
        output_dir.mkdir(parents = True, exist_ok = True)
        
        plot_confusion_matrices(results_path, output_dir)
        
        spindle_plot = output_dir / "confusion_spindle_5000rpm.png"
        linear_plot = output_dir / "confusion_linear.png"
        
        self.assertTrue(spindle_plot.exists(), "Spindle5000rpm confusion matrix plot not created")
        self.assertTrue(linear_plot.exists(), "Linear confusion matrix plot was not created")
        
        self.assertGreater(spindle_plot.stat().st_size, 0 ,"Spindle plot is empty")
        self.assertGreater(linear_plot.stat().st_size, 0 ,"Linear plot is empty")
        
    
    def test_plot_confusion_matrices_missing_file(self):
        fake_path = self.temp_dir / "no_file_here.csv"
        output_dir = self.temp_dir / "plots"
        
        try:
            plot_confusion_matrices(fake_path, output_dir)
        except Exception as e:
            self.fail(f"plot_confusion_matrices raised a missing file exception: {e}")
            
    
    def test_main(self):
        results_path = self.create_temp_results_csv("cli_resutls.csv")
        output_dir = self.temp_dir / "cli_plots"
        
        dummy_features = self.temp_dir / "features.csv"
        dummy_features.write_text("dummy, data\n1, 2\n")
        
        test_args = ["plotting.py", "-f", str(dummy_features), "-r", str(results_path), "-o", str(output_dir)]
        
        #temporarily mock sys.argv so argparse reads the above test args
        original_argv = sys.argv
        try:
            sys.argv = test_args
            main()
        finally:
            sys.argv = original_argv #restore sys.argv
            
        expected_plot = output_dir / "confusion_spindle_5000rpm.png"
        self.assertTrue(expected_plot.exists(), "CLI main execution failed to generate output plot.")
        self.assertGreater(expected_plot.stat().st_size, 0, "Generated plot file is empty!")
        

if __name__ == "__main__":
    unittest.main()
            
        

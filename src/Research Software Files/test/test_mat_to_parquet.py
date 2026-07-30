# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:01:15 2026

@author: map25bg
"""

""" 
This is for testing mat_to_parquet.py 
"""

import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mat_to_parquet import read_mat_file, mat_to_parquet, main

class TestMatToParquet(unittest.TestCase):
    
    def setUp(self):
        #temporary folder created before each test
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        
    def delete(self):
        #deletes the temporary folder
        self.temp_dir_obj.cleanup()
        
    def temporary_mat_file(self, filename="temporary.mat") -> Path:
        mat_path = self.temp_dir / filename
        with h5py.File(mat_path, "w") as f:
            struct_group = f.create_group("sensor_data")
            # Save simple numeric array signals inside
            struct_group.create_dataset("channel_1", data=np.array([1.0, 2.0, 3.0]))
            struct_group.create_dataset("channel_2", data=np.array([4.0, 5.0, 6.0]))
        return mat_path   
    
    #test read_mat_file
    def test_read_mat_file(self):
        mat_path = self.temporary_mat_file("test_read.mat")  #create temp file
        
        data_dict = read_mat_file(mat_path) #read the file
        
        self.assertIsInstance(data_dict, dict)
        self.assertIn("channel_1", data_dict)
        self.assertIn("channel_2", data_dict)
        np.testing.assert_array_equal(data_dict["channel_1"], [1.0, 2.0, 3.0])
     
    #test mat_to_parquet
    def test_mat_to_parquet(self):
        mat_path = self.temporary_mat_file("test_convert.mat")
        output_dir = self.temp_dir / "output"
              
        output_file = mat_to_parquet(mat_path, output_dir)  #run the function
        
        self.assertTrue(output_file.exists(), ".parquet file not created")
        self.assertEqual(output_file.name, "test_convert.parquet")
        
        df = pl.read_parquet(output_file)
        self.assertIn("channel_1", df.columns)
        self.assertEqual(len(df), 3) #check no data lost
     
    # test running cli
    def test_main(self):
        mat_path = self.temporary_mat_file("cli_test.mat")
        output_dir = self.temp_dir / "cli_output"
        
        test_args = ["mat_to_parquet.py", "-i", str(mat_path), "-o", str(output_dir)]
        
        #temporarily mock sys.argv so argparse reads the above test args
        original_argv = sys.argv
        try:
            sys.argv = test_args
            main()
        finally:
            sys.argv = original_argv #restore sys.argv
            
        expected_parquet = output_dir / "cli_test.parquet"
        self.assertTrue(expected_parquet.exists(), "CLI main() failed to create file.")

if __name__ == "__main__":
    unittest.main()
        
        
        
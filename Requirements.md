# Requirements

The code was produced using Python 3.12.11

## Required packages: 
- argparse
- pathlib
- numpy
- h5py
- polars
- sys
- scipy.stats
- tqdm
- scikit-learn
- matplotlib
- tempfile
- unittest

## By file:
***mat_to_parquet.py:*** argparse, pathlib, numpy, h5py, polars, sys

***get_features.py:*** argparse, pathlib, numpy, scipy.stats, polars, sys, tqdm

***classify.py:*** argprse, pathlib, pandas, scikit-learn

***plotting.py:*** argparse, pathlib, matplotlib, pandas, scikit-learn

### Test Required packages
***test_mat_to_parquet.py:*** sys, tempfile, unittest, pathlib, h5py, numpy, polars

***test_get_features.py:*** sys, tempfile, unittest, pathlib, numpy, polars

***test_classify.py:*** sys, tempfile, unittest, pathlib, numpy, pandas

***test_plotting.py:*** sys, tempfile, unittest, pathlib, pandas


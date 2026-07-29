# Data Science Coursework
**Author:** Beatrice Green

**Institution:** The University of Sheffield

**Module:** MAC4112

## Project Intentions

This project will use machine and tool sensor data to investigate machine and tool defect detection.
Taking data produced from experiemental trials that were replicating specific tool and machining defects and 
analysing it. Aiming to identify the expected defects and establish which are most readily identifiable. 

The defects were: 
- Baseline
- Misalignment (Machine bed tilt (0.27°, 0.27°, 0.32°))
- Surface cracks (drilling into the part in the cutting path before testing)
- Severe tool wear

And machine tool failure:
- Baseline
- Heavy tool
- Unbalanced tool
- Feed rate-adjusted (marginally reduced feed rate and spindle speed)

## The Dataset

The dataset used in this work is available from The University of Sheffield:

**Dominguez Caballero, J.A., Moore, J. and Stammers, J. (2023), _Sensor signals for machine tool and process health assessment_.**

DOI:

```text
https://doi.org/10.15131/shef.data.24125715.v1
```
The dataset is not available from this repository. The data is accepted as .mat files and then converted to parquet files by this code. 

## Code Structure

This code is intended to be used in the order set out below:
1. mat_to_parquet.py
    requires .mat files
    
2. get_features.py
    requires (-i) .parquet file or folder with .parquet file
    creates (-o) extracted_features.parquet
    
3. classify.py
    requires (-i) extracted_features.parquet 
    creates (-o) classification_results.csv and feature_importance.csv
    
4. plotting.py
    requires (-f) extracted_features.parquet and (-r) classification_results.csv 
    creates confusion matrices.png and scatter plots.png

Use the files created by the previous scripts to fulfill the input requirements of each script or use your equivalent.

Module requirements are laid out in Requirements.md

## How to

**mat_to_parquet.py:**
In PowerShell
1. Change directory to the scripts folder
```
cd "FILEPATH"
```
2. Run the script directly
```
python mat_to_parquet.py -i FILEPATH OF DATA FILE/FOLDER  -o FILEPATH OF OUTPUT FOLDER
```
In Spyder
    in the console:
```
%run mat_to_parquet.py -i "FILEPATH OF DATA FILE/FOLDER" -o "FILEPATH OF OUTPUT FOLDER"
```

**get_features.py:** 
Same as mat_to_parquet.py but change the input and output file locations

**classify.py:**
Same as mat_to_parquet.py change input and output locations and ensure that the input is a .parquet or .csv file - NOT a folder. 

**plotting.py:**
Requires two inputs (-f) feature_importance.csv and (-r) classification_results.csv.
Output (-o) is folder to save .png files




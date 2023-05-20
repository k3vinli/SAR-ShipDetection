# SAR-ShipDetection
ENEE439D Capstone Project

Ship Detection with SAR Imagery

# Useage Information
Install miniconda 

Create development environment by using 
```
conda env create -f environment.yml
```

Environment developed on CUDA12.0
CUDA versions 11.7 will not work since 11.8 version of PyTorch is installed.

# Directory Structure
project_dir/<br />
├─ Datasets/<br />
│  ├─ HRSID/<br />
│  │  ├─ annotations/<br />
│  │  │  ├─ test2017.json<br />
│  │  │  ├─ train2017.json<br />
│  │  │  ├─ train_test2017.json<br />
│  │  ├─ data/<br />
│  │  ├─ labels.json<br />
│  ├─ SSDD/<br />
├─ models/<br />
│  ├─ model_name/<br />
│  │  ├─ model_saved_weights/<br />
│  ├─ .ipynb file<br />
│  ├─ *any scripts*.py<br />
├─ environment.yml<br />

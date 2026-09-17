# MAP-Based Task-Oriented Precoding for Multiuser Communication

This repository contains the implementation of the paper:

> M. J. Ahmadi, Z. Zhang, R. F. Schaefer, and H. V. Poor, "MAP-Based Task-Oriented Precoding for Multiuser Communication," *IEEE Communications Letters*.

## Overview

The code implements the proposed MAP-based task-oriented communication framework, including feature extraction, task-oriented precoding, and MAP-based classification.

The implementation consists of two stages:

1. **Feature extraction and training in Python**
2. **Precoder optimization and MAP classification in MATLAB**

## Requirements

* Python with PyTorch (for feature generation)
* MATLAB (for precoding design, channel modeling, and classification accuracy evaluation)

## Dataset

The experiments use the **ModelNet40** dataset. The dataset itself is not included in this repository due to its size. Please download the dataset and place it under the `datasets/` directory inside `PythonFiles/`.

The required directory structure is:

```text
PythonFiles/
├── MAIN.py
├── models/
└── datasets/
    └── modelnet40_images_new_12x/
        ├── bed/
        ├── chair/
        ├── desk/
        ├── dresser/
        ├── guitar/
        ├── monitor/
        ├── sofa/
        ├── table/
        ├── toilet/
        └── xbox/
```

The experiments use the following 10 classes from ModelNet40:

`bed`, `chair`, `desk`, `guitar`, `dresser`, `monitor`, `sofa`, `table`, `xbox`, and `toilet`.

The other ModelNet40 classes are not used in the experiments.
```

## Running the Code

### 1. Configure and run the Python code

Before running the Python code, open `MAIN.py` and adjust the required system and training parameters, including:

* Number of workers/users $K$
* Feature dimension $D_k$
* Number of training epochs
* Batch size
* Learning rate
* Other required simulation parameters

Then run:

```bash
python3 MAIN.py
```

After running `MAIN.py`, select the desired feature extractor when prompted and press `Enter` to start training.

After the feature extractor has been trained for the specified number of epochs, the code generates the file:

```text
saved_MAT.mat
```

This file contains the trained feature-extraction parameters required by the MATLAB implementation.

### 2. Move the generated MAT file

Move or copy `saved_MAT.mat` to the directory containing the MATLAB file:

```text
main.m
```

### 3. Run the MATLAB code

Open MATLAB, navigate to the directory containing `main.m` and `saved_MAT.mat`, and run:

```matlab
main
```

In `main.m`, select the desired system and precoder parameters, including the precoding scheme, number of antennas, number of channel uses, transmit power, and other relevant parameters. Then run `main.m` to perform the precoder optimization and evaluate the resulting MAP classification accuracy.

## Reproducibility

To reproduce the results, first download and place the ModelNet40 dataset in the `datasets/` directory. Then configure the desired system and training parameters in `MAIN.py` and run the Python implementation to generate `saved_MAT.mat`. Move the generated file to the MATLAB directory, configure the desired parameters in `main.m`, and run the MATLAB implementation.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Ahmadi2026MAP,
  author  = {Ahmadi, Mohammad Javad and Zhang, Zhentian and Schaefer, Rafael F. and Poor, H. Vincent},
  title   = {MAP-Based Task-Oriented Precoding for Multiuser Communication},
  journal = {IEEE Communications Letters}
}
```

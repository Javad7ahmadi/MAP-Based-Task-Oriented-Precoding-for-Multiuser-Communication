# MAP-Based Task-Oriented Precoding for Multiuser Communication

This repository contains the implementation of the paper:

> M. J. Ahmadi, Z. Zhang, R. F. Schaefer, and H. V. Poor, "MAP-Based Task-Oriented Precoding for Multiuser Communication," *IEEE Communications Letters*.

## Overview

The code implements the proposed MAP-based task-oriented communication framework, including feature extraction, task-oriented precoding, and MAP-based classification.

The implementation consists of two stages:

1. **Feature extraction and training in Python**
2. **Precoder optimization and MAP classification in MATLAB**

## Requirements

- Python with PyTorch
- MATLAB
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

For example:

```text
MATLAB/
├── main.m
├── saved_MAT.mat
└── ...
```

### 3. Run the MATLAB code

Open MATLAB, navigate to the directory containing `main.m` and `saved_MAT.mat`, and run:

```matlab
main
```

The MATLAB code loads the trained feature extractor from `saved_MAT.mat`, selects the specified precoding scheme, performs the precoder optimization, and evaluates the resulting MAP classification accuracy.

## Reproducibility

To reproduce the results, first configure the desired system and training parameters in `MAIN.py`. Run the Python implementation to generate `saved_MAT.mat`, then place the generated file in the MATLAB directory and run `main.m`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Ahmadi2026MAP,
  author  = {Ahmadi, Mohammad Javad and Zhang, Zhentian and Schaefer, Rafael F. and Poor, H. Vincent},
  title   = {MAP-Based Task-Oriented Precoding for Multiuser Communication},
  journal = {IEEE Communications Letters}
}
```

[Readme.txt](https://github.com/user-attachments/files/29289870/Readme.txt)

# MAP-Based Task-Oriented Precoding for Multiuser Communication

This repository implements the methods proposed in the paper:

**MAP-Based Task-Oriented Precoding for Multiuser Communication**
Submitted to IEEE Communications Letters

The code includes the proposed MAP-based precoding and feature learning framework, as well as the baseline MCR² method. MATLAB scripts are provided to reproduce all figures in the paper.

---

## 1. Requirements

Python 3.8 or higher is required.

Install the required packages:
pip install torch numpy tensorboard

MATLAB is also required to generate the figures.

---

## 2. Feature Extractor Configuration

Before training, open Config.py inside the feature_extractor folder and adjust the neural network parameters such as architecture, embedding dimension, and training settings according to your setup.

---

## 3. Training

Run:
python Main.py

During execution, you will be asked to choose a mode:

1 → MCR² baseline method
2 → Proposed MAP-based feature extractor and precoding method

You must run the training twice:
First with option 1, then again with option 2.

Wait until each training run is fully completed before starting the next one.

---

## 4. Export Results

After training, run:
python Export.py

Again, select:
1 for MCR² baseline
2 for proposed method

This will generate .mat files for MATLAB processing.

---

## 5. Plot Results

In MATLAB:

Run Figure1.m to generate Figure 1 of the paper.

Run Figure2.m to generate Figure 2 of the paper.

---

## 6. Citation

If you use this code, please cite:

M. J. Ahmadi, R. F. Schaefer, and H. V. Poor,
"MAP-Based Task-Oriented Precoding for Multiuser Communication,"
arXiv preprint, 2026, submitted to IEEE Communications Letters.

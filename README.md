# Deep Learning for Perception 
### Building, Breaking and Fixing a Neural Network

National University of Computer and Emerging Sciences — Fall 2026

## Overview

This notebook builds a feedforward neural network end to end on Fashion-MNIST, deliberately
pushes it into overfitting, then repairs it using regularisation and hyperparameter tuning —
measuring the effect of every design choice.

## Contents

| Part | Description | Marks |
|---|---|---|
| Setup | Load, normalise, flatten, 80/20 train/validation split | — |
| Part 1 | Two-layer MLP implemented from scratch in NumPy; gradients verified against PyTorch | 15 |
| Part 2 | Baseline model and activation function study (sigmoid, tanh, ReLU, leaky ReLU) | 15 |
| Part 3 | Loss function comparison (cross-entropy vs MSE) + tabular regression task | 10 |
| Part 4 | Optimiser comparison (SGD, SGD+Momentum, RMSProp, Adam) | 10 |
| Part 5 | Deliberately forcing overfitting on a large network / small dataset | 10 |
| Part 6 | Regularisation study: L2, L1, dropout, batch norm, early stopping, augmentation, more data | 25 |
| Part 7 | Random search + 5-fold cross-validation hyperparameter tuning, final test evaluation | 15 |

Every code cell has a short markdown note above it explaining what that cell does, so the
notebook can be followed top to bottom without needing to read the code first.

## Dataset

[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) — loaded via
the Kaggle dataset CSVs (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`).

- 43 duplicate rows removed from the training set, 1 from the test set
- Pixel values normalised to [0, 1], images flattened to 784-dimensional vectors
- Training data split 80/20 (stratified) into train and validation sets
- The provided test set was kept untouched until the final evaluation in Part 7

## Key Results

| Metric | Part 2 Baseline | Final Tuned Model |
|---|---|---|
| Accuracy | 88.97% (validation) | 90.36% (test) |
| Macro Precision | — | 90.32% |
| Macro Recall | — | 90.36% |
| Macro F1 | — | 90.29% |

**Selected configuration (Part 7):** learning rate = 0.0005, hidden width = 512,
dropout = 0.2, selected via random search (12 configurations) scored with 5-fold
cross-validation (mean CV accuracy 84.88%), then retrained on the full training set.

**Most impactful change:** increasing training data from 2,000 to 20,000 samples reduced
the train/validation accuracy gap from 17.25 to 10.18 percentage points, while training
accuracy dropped only slightly (100% → 98.26%) — a better trade-off than any explicit
regularisation technique tested.

## How to Reproduce

1. Open the notebook on [Kaggle](https://www.kaggle.com/) with a **GPU T4 x2** accelerator.
2. Add the [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist) as a notebook input.
3. Run all cells from top to bottom (`Run All`). All random seeds are fixed (`random_state=42` /
   `np.random.seed(42)`), so the results are reproducible.
4. No other files are needed — the only external data used besides Fashion-MNIST is
   `sklearn.datasets.fetch_california_housing`, which downloads automatically in Part 3.


## Notes

- Categorical cross-entropy loss uses one-hot encoded labels throughout, for consistency
  between Part 1 (NumPy) and the later Keras-based parts.
- The validation set (not the test set) was used for every intermediate model-selection
  decision in Parts 2–6, to avoid test-set leakage. The test set was evaluated exactly
  once, in Part 7.

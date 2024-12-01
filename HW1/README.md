# HW1: FashionMNIST x LeNet-5

## Overview
This project implements the LeNet-5 architecture over the FashionMNIST dataset and compares the effects of different regularization techniques:
- **No Regularization**
- **Dropout** (at the hidden layer)
- **Weight Decay** (L2 regularization)
- **Batch Normalization**

The project includes scripts for training the model, generating convergence graphs, and comparing the techniques.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

---

## Requirements
This project uses `Python 3.11.8`. All dependencies are managed with [Poetry](https://python-poetry.org/). Poetry files 'pyprject.toml' and 'poetry.lock' files are included.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/devleibowitz/DeepLearningHWs.git
   cd DeepLearningHWs/HW1/

## Usage 
Run the following command to train and evaluate the models:
   ```bash
   lenet5.ipynb
   ```
The notebook will train all four variants of the LeNet5 model.

## Results
The above code generates plots showing each model's training and testing accuracies in the results folder. The trained models are also saved here as .pth files.

### LeNet5 Without Regularization
This model is the basic implemenation of Lenet5 without any modifications. The red vertical line shows where the best model checkpoint was taken.
![No Regularization](results/No%20Regularization.png)

### LeNet5 with Dropout
In this version, dropout layer added to avoid overfitting. Dropout ignores a given percentage of neurons at training and thereby prevents the network from overemphasizing specific signals. We can tell from the plot here that the training accuracy is indeed less prone to overfitting here than the original version above. We also ultimately reach a better test accuracy altogether.
![Dropout Regularization](results/Dropout%20Regularization.png)

### LeNet5 with Batch Normalization
Batch normalization reduces internal covraite shift by normalizing inputs and therefore helps the optimization procecess by maintaining gradients within a regular range. This model version reached the best accuracy of all.
![Batch Normalization Regularization](results/Batch%20Normalization%20Regularization.png)


### LeNet5 with Weight Decay
The model implements L2 regularization to penalize large weights and thereby reduce overfitting. Regularization reduces the effect of noise in the training data and thereby enforces the model to optimize better. This model performs best in terms of minimal overfitting- we can see the training and test accuracies are more closely aligned than the others.
![Weight Decay Regularization](results/Weight%20Decay%20Regularization.png)

### Table Summary of Results:

| Technique             | Train Accuracy (%) | Test Accuracy (%) |
|-----------------------|--------------------|-------------------|
| No Regularization     | 97.87              | 90.00             |
| Dropout               | 91.59              | 90.05             |
| Weight Decay (L2)     | 93.27              | 91.00             |
| Batch Normalization   | 98.60              | 89.95             |

You can download the results in a csv file here:
[LeNet5 Results](results/lenet5_results.csv)

# Multimodal Deep Learning for Fetal Distress Prediction

This repository contains the code and resources for a research project focused on predicting fetal distress using a multimodal deep learning framework. The approach combines structured maternal and neonatal clinical features with unstructured CTG signal data (fetal heart rate and uterine contractions) to improve the accuracy of perinatal risk assessment.

## 📋 Overview

Stillbirth and birth-related complications remain a leading cause of neonatal mortality worldwide. Cardiotocography (CTG) is widely used to monitor fetal heart rate (FHR) and uterine contractions (UC), but its interpretation is subjective and error-prone. This project explores a machine learning pipeline that integrates:
- **Structured features** from Electronic Health Records (EHR)
- **Unstructured time-series signals** from CTG monitoring
- **Fusion model** combining both modalities for optimal prediction

## 🧠 Models Implemented
- **LightGBM** for structured EHR data
- **Transformer-based model** for raw CTG signals
- **Mamba**: State-space model for sequential signal modeling
- **Timer & Chronos-Bolt**: Pretrained time-series transformers (exploratory)
- **Fusion Model**: Combines LightGBM and Transformer outputs using late fusion

## 📁 Directory Structure

├── data/ # Preprocessed dataset files (EHR & CTG) ├── notebooks/ # Jupyter notebooks for model experiments and visualizations ├── models/ # Saved models and weights ├── figures/ # All result figures and diagrams ├── scripts/ │ ├── train_lightgbm.py # LightGBM training and evaluation │ ├── train_transformer.py │ ├── train_mamba.py │ ├── fusion_model.py │ └── preprocessing.py # Signal cleaning, EHR formatting ├── results/ │ └── report.pdf # Final report (with figures, tables, and references) ├── requirements.txt └── README.md


## 🚀 Quick Start

1. **Install dependencies**  
This project uses `Python 3.11.8`. All dependencies are managed with [Poetry](https://python-poetry.org/). Poetry files 'pyprject.toml' and 'poetry.lock' files are included.
2. **Prepare the data**
Ensure the dataset (CTU-UHB) is downloaded and placed in data/. Preprocessing scripts will clean and format the data automatically.
3. **Train LightGBM on EHR Data**
   ```bash
    python scripts/train_lightgbm.py data/ehr_features.csv

4. **Train Transformer on CTG Data**
   ```bash
    python scripts/train_transformer.py

5. **Run the Fusion Model**
   ```bash
    python scripts/fusion_transformer.py

## 📊 Results

| Model                   | Accuracy | Precision | Recall | F1-Score | AUROC |
|------------------------|----------|-----------|--------|----------|--------|
| LightGBM (EHR only)    | 0.66     | 0.52      | 0.83   | 0.64     | 0.71   |
| Transformer (CTG only) | 0.68     | 0.61      | 0.76   | 0.67     | 0.74   |
| Mamba (CTG only)       | 0.63     | 0.58      | 0.62   | 0.60     | 0.68   |
| Chronos-Bolt (CTG only)| 0.60     | 0.50      | 0.58   | 0.54     | 0.65   |
| **Fusion (Transformer + EHR)** | **0.74** | **0.70** | **0.81** | **0.72** | **0.80** |

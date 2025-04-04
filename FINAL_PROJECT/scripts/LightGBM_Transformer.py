import torch
import torch.nn as nn
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
from joblib import dump
import pandas as pd

import pandas as pd
import numpy as np
import torch

# Load structured features
ehr_df = pd.read_csv('data/processed/labeled_ehr.csv')  # includes 'label' column?
labels = pd.read_csv('data/processed/phenotype_2_labels.csv').values.squeeze()  # (n_samples,)

# Drop label column if it's already in ehr_df to avoid duplication
if 'label' in ehr_df.columns:
    ehr_data = ehr_df.drop(columns=['label', 'id']).values
else:
    ehr_data = ehr_df.values

# Load CTG tensor (PyTorch format)
ctg_data = torch.load('data/processed/fhr_uc_signal_tensor.pt').numpy()  # shape: (n_samples, seq_len, 2)

# Split data
X_ehr_train, X_ehr_test, X_ctg_train, X_ctg_test, y_train, y_test = train_test_split(
    ehr_data, ctg_data, labels, test_size=0.2, random_state=42
)

# Load pretrained models
lightgbm_model = joblib.load('models/lightgbm_model.pkl')          # LightGBM model
transformer_model = torch.load('models/transformer_model.pt')      # PyTorch Transformer
transformer_model.eval()

# Function to extract LightGBM leaf embeddings or probabilities
def get_lightgbm_outputs(model, X):
    return model.predict_proba(X)  # shape: (n_samples, 2) — we’ll use [:,1] for positive class

# Function to extract Transformer latent outputs
def get_transformer_outputs(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        output = model(X_tensor)  # assumed shape: (n_samples, latent_dim)
        return output.numpy()

# Get model outputs for fusion
lgbm_probs_train = get_lightgbm_outputs(lightgbm_model, X_ehr_train)[:, 1].reshape(-1, 1)
lgbm_probs_test = get_lightgbm_outputs(lightgbm_model, X_ehr_test)[:, 1].reshape(-1, 1)

transformer_latent_train = get_transformer_outputs(transformer_model, X_ctg_train)
transformer_latent_test = get_transformer_outputs(transformer_model, X_ctg_test)

# Concatenate features for fusion
fusion_train = np.concatenate([lgbm_probs_train, transformer_latent_train], axis=1)
fusion_test = np.concatenate([lgbm_probs_test, transformer_latent_test], axis=1)

# Train final fusion classifier
fusion_head = LogisticRegression(max_iter=1000, class_weight='balanced')
fusion_head.fit(fusion_train, y_train)

# Predict and evaluate
y_pred = fusion_head.predict(fusion_test)
y_prob = fusion_head.predict_proba(fusion_test)[:, 1]

print("=== Fusion Model Evaluation ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUROC:", roc_auc_score(y_test, y_prob))



# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the trained fusion head model
dump(fusion_head, 'models/fusion_head.joblib')

print("Fusion head model saved to models/fusion_head.joblib")
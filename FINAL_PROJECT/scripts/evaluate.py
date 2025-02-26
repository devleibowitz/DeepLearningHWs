import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

from load_data import prepare_data
from train_LSTM import LSTMClassifier  # Import the model definition

# Set the root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

# Set other directories
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def load_model(model_path, input_size, hidden_size, num_layers, num_classes, device):
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Assuming binary classification

    test_loss /= len(test_loader)
    return test_loss, all_predictions, all_labels, all_probabilities

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{MODEL_DIR}/figures/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{MODEL_DIR}/figures/roc_curve.png')
    plt.close()

if __name__ == "__main__":
    # Data paths
    data_path = os.path.join(DATA_DIR, "processed", "ctg_tensor.pt")
    labels_path = os.path.join(DATA_DIR, "processed", "labels.csv")
    model_path = os.path.join(ROOT_DIR, "models", "lstm_model.pth")

    print(f"Root Directory: {ROOT_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Data Path: {data_path}")
    print(f"Labels Path: {labels_path}")
    

    # Hyperparameters (should match those used in training)
    input_size = 21620
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    batch_size = 32

    # Prepare data (assuming prepare_data returns test_loader as well)
    _, _, test_loader = prepare_data(data_path, labels_path, batch_size, include_test=True)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, input_size, hidden_size, num_layers, num_classes, device)

    # Evaluate the model
    criterion = nn.CrossEntropyLoss()
    test_loss, all_predictions, all_labels, all_probabilities = evaluate_model(model, test_loader, criterion, device)

    # Print evaluation metrics
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, classes=['Class 0', 'Class 1'])

    # Plot ROC curve
    plot_roc_curve(all_labels, all_probabilities)

    print("Evaluation completed. Check the output directory for confusion matrix and ROC curve plots.")

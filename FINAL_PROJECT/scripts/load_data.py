import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class CTGDataset(Dataset):
    def __init__(self, data_tensor, labels_csv):
        self.data = data_tensor
        self.labels = pd.read_csv(labels_csv)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.tensor(self.labels.iloc[idx]['label'], dtype=torch.long)
        return sample, label

def load_data(data_path, labels_path):
    """
    Load the data tensor and labels CSV file.
    """
    data_tensor = torch.load(data_path)
    labels_df = pd.read_csv(labels_path)
    return data_tensor, labels_df

def preprocess_data(data_tensor):
    """
    Preprocess the data tensor. This function can be expanded based on specific needs.
    """
    # Normalize the data
    scaler = StandardScaler()
    data_reshaped = data_tensor.reshape(-1, data_tensor.shape[-1])
    data_normalized = scaler.fit_transform(data_reshaped)
    data_tensor_normalized = torch.tensor(data_normalized.reshape(data_tensor.shape), dtype=torch.float32)
    
    return data_tensor_normalized

def prepare_data(data_path, labels_path, batch_size=32, train_split=0.7, val_split=0.15, include_test=False):
    """
    Main function to prepare the data.
    """
    # Load data
    data_tensor, labels_df = load_data(data_path, labels_path)
    
    # Preprocess data
    data_tensor_preprocessed = preprocess_data(data_tensor)
    
    # Create dataloaders
    if include_test:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_tensor_preprocessed, 
            labels_path, 
            batch_size, 
            train_split, 
            val_split
        )
        return train_loader, val_loader, test_loader
    else:
        train_loader, val_loader = create_dataloaders(
            data_tensor_preprocessed, 
            labels_path, 
            batch_size, 
            train_split / (train_split + val_split)
        )
        return train_loader, val_loader

def create_dataloaders(data_tensor, labels_csv, batch_size=32, train_split=0.7, val_split=0.15):
    """
    Create train, validation, and optionally test DataLoaders.
    """
    dataset = CTGDataset(data_tensor, labels_csv)
    
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Current working directory:", os.getcwd())

    # Define the data directory relative to the project root
    DATA_DIR = "data"

    # Example usage
    data_path = f"{DATA_DIR}/processed/ctg_tensor.pt"
    labels_path = f"{DATA_DIR}/processed/labels.csv"
    
    train_loader, val_loader = prepare_data(data_path, labels_path)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Check the shape of a batch
    for batch_data, batch_labels in train_loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break

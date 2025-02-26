import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

# Load your data
data_tensor = torch.load('your_data_tensor.pt')  # Load your torch.Size([552, 2, 21620]) tensor
dataset = CTGDataset(data_tensor, 'labels.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

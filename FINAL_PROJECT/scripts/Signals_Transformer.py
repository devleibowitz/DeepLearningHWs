import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig
import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="bleach")


DATA_DIR = "/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data"

# Load Data
def load_data(label_path, tensor_path, mapping_path):
    labels_df = pd.read_csv(label_path)  # Load CSV labels
    signal_tensor = torch.load(tensor_path)  # Load FHR + UC tensor
    with open(mapping_path, 'r') as f:
        signal_id_mapping = json.load(f)  # Load ID mappings
    return labels_df, signal_tensor, signal_id_mapping

# Custom Dataset
class FHRDataset(Dataset):
    def __init__(self, signal_tensor, labels_df, window_size=512, overlap=256):
        self.window_size = window_size
        self.step = window_size - overlap
        self.samples = []
        
        labels = labels_df['label'].values
        for signal_idx in range(len(signal_tensor)):
            signal = signal_tensor[signal_idx]
            label = labels[signal_idx]
            num_windows = (signal.size(0) - window_size) // self.step + 1
            for i in range(num_windows):
                start = i * self.step
                self.samples.append((
                    signal[start:start+window_size],
                    torch.tensor(label, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Transformer Model
class SignalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2, model_name="bert-base-uncased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.input_proj = nn.Linear(input_dim, self.config.hidden_size)  # Critical fix
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.input_proj(x)  # Project to hidden_size
        outputs = self.encoder(inputs_embeds=x).last_hidden_state
        logits = self.fc(outputs.mean(dim=1))  # Average pooling instead of CLS token
        return logits


# Sliding Window Prediction
def sliding_window_predict(model, signal, window_size=512, overlap=256, device='cuda'):
    model.eval()
    seq_len = signal.size(0)
    step = window_size - overlap
    logits = []
    
    for start in range(0, seq_len, step):
        end = start + window_size
        window = signal[start:end]
        if window.size(0) < window_size:  # Handle edge case
            window = F.pad(window, (0, 0, 0, window_size - window.size(0)))
        window = window.unsqueeze(0).to(device)
        with torch.no_grad():
            logits.append(model(window))
    
    avg_logits = torch.mean(torch.cat(logits), dim=0)
    return torch.argmax(avg_logits)


# Training Loop
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch in train_loader:
        inputs, labels = batch  # Properly unpack the batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

from torch.utils.tensorboard import SummaryWriter

def train_model_with_tensorboard(model, train_loader, criterion, optimizer, device, num_epochs=10):
    writer = SummaryWriter()  # Initialize TensorBoard writer
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log loss to TensorBoard
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

        writer.add_scalar('Epoch Avg Loss', total_loss / len(train_loader), epoch)

    writer.close()

# Main Execution
if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load arguments
    label_path = os.path.join(DATA_DIR, "processed", sys.argv[1])
    tensor_path = os.path.join(DATA_DIR, "processed", sys.argv[2])
    mapping_path = os.path.join(DATA_DIR, "processed", sys.argv[3])

    # Load data
    labels_df, signal_tensor, signal_id_mapping = load_data(label_path, tensor_path, mapping_path)

    # Create Dataset & DataLoader
    dataset = FHRDataset(signal_tensor, labels_df, window_size=512, overlap=256)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Transformer model
    model = SignalTransformer(input_dim=signal_tensor.shape[-1]).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Train model
    print("Starting training...")
    # train_model(model, train_loader, criterion, optimizer, device)
    train_model_with_tensorboard(model, train_loader, criterion, optimizer, device)

    all_preds = []
    for signal in signal_tensor:
        pred = sliding_window_predict(model, signal, device=device)
        all_preds.append(pred.cpu())

    # Predict using sliding window
    # predictions = sliding_window_predict(model, signal_tensor, device=device)
    # print("Predictions:", predictions.cpu().numpy())

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add the project root to the Python path
# ROOT_DIR = os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# MODEL_DIR = f'{ROOT_DIR}/models'


# while not os.path.exists(os.path.join(ROOT_DIR, 'data')):
#     ROOT_DIR = os.path.dirname(ROOT_DIR)
#     if ROOT_DIR == os.path.dirname(ROOT_DIR):  # Reached the system root
#         raise FileNotFoundError("Could not find the 'data' directory in any parent directory.")
# sys.path.append(ROOT_DIR)

from load_data import prepare_data

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(f'{MODEL_DIR}/figures/LSTM_loss_plot.png')
    plt.close()

if __name__ == "__main__":
    # Data paths
    # Set the root directory
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(ROOT_DIR)

    # Set other directories
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    # Data paths
    data_path = os.path.join(DATA_DIR, "processed", "ctg_tensor.pt")
    labels_path = os.path.join(DATA_DIR, "processed", "labels.csv")

    print(f"Root Directory: {ROOT_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Data Path: {data_path}")
    print(f"Labels Path: {labels_path}")
    
    # Hyperparameters
    input_size = 21620 # Batch data shape: torch.Size([32, 2, 21620]), Batch labels shape: torch.Size([32])
    hidden_size = 64
    num_layers = 2
    num_classes = 2  # Assuming binary classification, adjust if needed
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32
    
    # Prepare data
    train_loader, val_loader = prepare_data(data_path, labels_path, batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Plot losses
    plot_losses(train_losses, val_losses)
    
    # Save the model
    torch.save(model.state_dict(), f'{MODEL_DIR}/lstm_model.pth')
    print("Model saved successfully.")

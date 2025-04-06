import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from mamba_ssm import Mamba
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR

class MambaNextToken(nn.Module):
    def __init__(self, d_model=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, d_model, seq_len) → (batch, seq_len, d_model)
        x = self.mamba(x)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x
    

class MambaBinaryClassifier(nn.Module):
    def __init__(self, d_model=2, d_state=16, d_conv=4, expand=2, dropout_rate=0.3, linear_input_dim = 21620):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Final classifier
        self.proj = nn.Linear(linear_input_dim, 512)  # Reduce dimensions first
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, d_model, seq_len) → (batch, seq_len, d_model)
        x = self.mamba(x)  # (batch, seq_len, d_model)

        x = x[:, :, 0]  # Select the first channel (batch, seq_len)

        x = self.proj(x)
        x = self.dropout(x)  # Apply dropout before classification
        x = self.classifier(x)
        return torch.sigmoid(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for Linear layers
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Bias initialized to zero
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # He initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Bias initialized to zero

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, scheduler, device, num_epochs=40):
    writer = SummaryWriter()

    """Train and evaluate function"""
    model.train()

    # Freeze 'mamba' layer initially
    for param in model.mamba.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        total_train_loss, total_train_accuracy = 0, 0
        total_test_loss, total_test_accuracy = 0, 0
        all_train_labels, all_train_preds = [], []
        all_test_labels, all_test_preds = [], []

        if epoch == 6:
            for param in model.mamba.parameters():
                param.requires_grad = True

        # Training
        model.train()
        for inputs, labels in train_loader:
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Warning: Input contains NaNs or Infs!")
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            accuracy = (outputs.round() == labels.unsqueeze(1).float()).float().mean()
            total_train_accuracy += accuracy.item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(outputs.cpu().detach().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_accuracy / len(train_loader)

        # Compute sensitivity, specificity, AUC for training
        train_preds = [1 if x >= 0.5 else 0 for x in all_train_preds]
        cm = confusion_matrix(all_train_labels, train_preds)
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        auc = roc_auc_score(all_train_labels, all_train_preds)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0


        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
        writer.add_scalar("Sensitivity/train", sensitivity, epoch)
        writer.add_scalar("Specificity/train", specificity, epoch)
        writer.add_scalar("AUC/train", auc, epoch)
        writer.add_scalar("Precision/train", precision, epoch)

        # Testing
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                total_test_loss += loss.item()
                accuracy = (outputs.round() == labels.unsqueeze(1).float()).float().mean()
                total_test_accuracy += accuracy.item()


                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(outputs.cpu().detach().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_accuracy / len(test_loader)
               # Compute sensitivity, specificity, AUC for testing
        # Check if cm has the expected shape
        if len(all_test_labels) > 0 and len(np.unique(all_test_labels)) > 1:
            test_preds = [1 if x >= 0.5 else 0 for x in all_test_preds]
            cm = confusion_matrix(all_test_labels, test_preds)

            # Calculate sensitivity and specificity only if cm has the correct shape
            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            auc = roc_auc_score(all_test_labels, all_test_preds)
        else:
            print("len(all_test_labels)", len(all_test_labels), "len(np.unique(all_test_labels)",len(np.unique(all_test_labels)))
            sensitivity, specificity, auc = 0, 0, 0  # Set metrics to 0 if cm is invalid

        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test", avg_test_acc, epoch)
        writer.add_scalar("Sensitivity/test", sensitivity, epoch)
        writer.add_scalar("Specificity/test", specificity, epoch)
        writer.add_scalar("AUC/test", auc, epoch)
        writer.add_scalar("Precision/train", precision, epoch)


        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        scheduler.step()
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

    writer.flush()
    return writer
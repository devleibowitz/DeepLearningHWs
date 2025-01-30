import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# Set random seeds for reproducibility
np.random.seed(205917883)
torch.manual_seed(305134496)

FAHSION_MNIST_CLASSES = datasets.FashionMNIST.classes
FAHSION_MNIST_N_CLASSES = len(datasets.FashionMNIST.classes)
def get_fashion_mnist_datasets(num_labels):
    

    # Load Fashion MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create labeled and unlabeled subsets
    labeled_indices = []
    unlabeled_indices = []

    for class_id in range(FAHSION_MNIST_N_CLASSES):
        class_indices = torch.argwhere(train_dataset.targets == class_id)
        labeled_indices.extend((class_indices[torch.randperm(len(class_indices))[:num_labels // FAHSION_MNIST_N_CLASSES]]).squeeze().tolist())

    unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))
    labeled_subset = Subset(train_dataset, labeled_indices)
    unlabeled_subset = Subset(train_dataset, unlabeled_indices)

    labeled_loader = DataLoader(labeled_subset, batch_size=128, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return labeled_loader, unlabeled_loader, test_loader
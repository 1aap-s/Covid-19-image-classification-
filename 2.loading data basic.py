#DATA LOADING VIA TORCH 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize all images
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset
dataset = datasets.ImageFolder(root="dataset/train", transform=transform)#path of the dataset


# Split into train + validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Test dataset (unchanged)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)#path of the dataset

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))
print(dataset.classes)
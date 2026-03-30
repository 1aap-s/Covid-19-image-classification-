import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Same transforms as before
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
dataset = datasets.ImageFolder(root="train", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get one batch
images, labels = next(iter(loader))

# Convert tensor → image for display
images = images * 0.5 + 0.5   # unnormalize

# Plot images
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = images[i].permute(1, 2, 0)  # CHW → HWC
    plt.imshow(img)
    plt.title(labels[i].item())
    plt.axis('off')

plt.show()
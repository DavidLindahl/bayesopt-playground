import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from data.data_loader import load_MNIST
from model.CNN_model import CNN, train
# Load data
train_loader, val_loader, test_loader = load_MNIST(
    train_size=500, test_size=100, val_size=100, batch_size=32
)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the device
model = CNN(32, 32, 3, 5, 2, 0.25, 128).to(device)

# Create an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for a small number of epochs (adjust as needed for testing)
num_epochs = 10
test_accuracy = train(model, device, train_loader, test_loader, optimizer, num_epochs)

print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        conv_nodes_1, conv_nodes_2, kernel_size_1, kernel_size_2, maxpool_size, dropout_rate, fc_nodes
        # First convolution: 1 input channel (grayscale), 32 output channels, 3x3 kernel, padding to preserve spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_nodes_1, kernel_size=kernel_size_1, padding=1)
        # Second convolution: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(conv_nodes_1, conv_nodes_2, kernel_size=kernel_size_2, padding=1)
        
        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # After two poolings, the 28x28 image becomes 7x7 (since 28 -> 14 -> 7)
        self.fc1 = nn.Linear(64 * 7 * 7, fc_nodes)
        self.fc2 = nn.Linear(fc_nodes, 10)  # 10 output classes for MNIST

    def forward(self, x):
        # Input x shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))  # -> (batch_size, 32, 28, 28)
        x = self.pool(x)           # -> (batch_size, 32, 14, 14)
        x = F.relu(self.conv2(x))  # -> (batch_size, 64, 14, 14)
        x = self.pool(x)           # -> (batch_size, 64, 7, 7)
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)           # Output logits for each of the 10 classes
        return x

# Example usage:
# model = MNIST_CNN()
# output = model(torch.randn(64, 1, 28, 28))  # Dummy batch of 64 images

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and targets to the specified device (CPU or GPU)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()      # Clear gradients from the previous iteration
        output = model(data)       # Forward pass: compute predicted outputs
        loss = F.cross_entropy(output, target)  # Compute the loss
        loss.backward()            # Backward pass: compute gradients
        optimizer.step()           # Update model parameters
        
        # Print training status every 'log_interval' batches
        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} '
                f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item():.6f}'
            )

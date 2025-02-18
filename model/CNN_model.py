import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        conv_nodes_1,
        conv_nodes_2,
        kernel_size_1,
        kernel_size_2,
        maxpool_size,
        dropout_rate,
        fc_nodes,
    ):
        super(CNN, self).__init__()
        # First convolution: 1 input channel (grayscale), 32 output channels, 3x3 kernel, padding to preserve spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_nodes_1,
            kernel_size=kernel_size_1,
            padding=int((kernel_size_1-1)/2),
        )

        # Second convolution: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(
            in_channels=conv_nodes_1,
            out_channels=conv_nodes_2,
            kernel_size=kernel_size_2,
            padding=int((kernel_size_2-1)/2),
        )
        self.conv_2 = conv_nodes_2
        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # After two poolings, the 28x28 image becomes 7x7 (since 28 -> 14 -> 7)
        self.fc1 = nn.Linear(conv_nodes_2 * 7 * 7, fc_nodes)
        self.fc2 = nn.Linear(fc_nodes, 10)  # 10 output classes for MNIST

    def forward(self, x):
        # Input x shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))  # -> (batch_size, 32, 28, 28)
        x = self.pool(x)  # -> (batch_size, 32, 14, 14)
        x = F.relu(self.conv2(x))  # -> (batch_size, 64, 14, 14)
        x = self.pool(x)  # -> (batch_size, 64, 7, 7)
        x = self.dropout(x)
        x = x.view(-1, self.conv_2 * 7 * 7)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits for each of the 10 classes
        return x


# Example usage:
# model = MNIST_CNN()
# output = model(torch.randn(64, 1, 28, 28))  # Dummy batch of 64 images


def train(model, device, train_loader, test_loader, optimizer, num_epochs):
    """
    Trains the model for a given number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): Device on which to perform training (CPU or GPU).
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
        num_epochs (int): Number of epochs to train.
        log_interval (int, optional): Interval (in batches) to log training status.
        test_loader (DataLoader): The model is evaluated on test data after training.
    """

    def evaluate(model, device, test_loader):
        """
        Evaluates the model on test data and returns the accuracy.

        Args:
            model (torch.nn.Module): The trained model.
            device (torch.device): Device on which to perform evaluation.
            test_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        model.eval()  # Set the model to evaluation mode
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Sum up batch loss (using sum to later compute the average loss)
                # Get the index of the max log-probability (predicted class)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)

        return accuracy

    model.train()  # Set the model to training mode
    for epoch in range(1, num_epochs + 1):

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            output = model(data)  # Forward pass
            loss = F.cross_entropy(output, target)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

    # If a test_loader is provided, evaluate the model after each epoch
    accuracy = evaluate(model, device, test_loader)
    return accuracy

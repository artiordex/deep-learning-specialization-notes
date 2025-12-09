"""Simple neural network example using PyTorch.

This module demonstrates a basic neural network implementation for classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    """A simple feedforward neural network.
    
    Architecture:
        - Input layer
        - Hidden layer with ReLU activation
        - Output layer with softmax
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """Initialize the neural network.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            num_classes: Number of output classes
        """
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ConvNet(nn.Module):
    """A simple convolutional neural network for image classification.
    
    Architecture:
        - 2 convolutional layers with max pooling
        - 2 fully connected layers
    """
    
    def __init__(self, num_classes: int = 10):
        """Initialize the CNN.
        
        Args:
            num_classes: Number of output classes
        """
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # For 28x28 input images
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        out = self.relu(self.conv1(x))
        out = self.pool(out)
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def example_training():
    """Example of training a simple neural network.
    
    This function demonstrates:
        - Creating synthetic data
        - Initializing a model
        - Setting up loss and optimizer
        - Training loop
    """
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    num_classes = 3
    num_samples = 1000
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 5
    
    # Create synthetic data
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    print('Training complete!')


if __name__ == '__main__':
    print("Simple Neural Network Example")
    print("-" * 40)
    example_training()

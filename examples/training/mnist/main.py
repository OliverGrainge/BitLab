"""Simple MNIST training example with BitLinear layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bitlab.bnn import Module, BitLinear


class SimpleMLP(Module):
    """Simple MLP with BitLinear layers for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.layer1 = BitLinear(input_size, hidden_size)
        self.layer2 = BitLinear(hidden_size, hidden_size)
        self.layer3 = BitLinear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='../../data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Print loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / num_batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
        
        # Print epoch summary
        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] completed - Average Loss: {avg_loss:.4f}")
        print("-" * 50)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()


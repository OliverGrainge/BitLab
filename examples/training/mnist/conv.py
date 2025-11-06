"""Simple MNIST training example with BitConv2d layers."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bitlab.bnn import Module, BitLinear, BitConv2d


class SimpleCNN(Module):
    """Simple CNN with BitConv2d layers for MNIST classification."""
    
    def __init__(self, num_classes=10, quant_type="ai8pc_wpt"):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = BitConv2d(32, 64, kernel_size=3, padding=1, quant_type=quant_type)  # 28x28 -> 28x28
        self.conv3 = BitConv2d(64, 64, kernel_size=3, stride=2, padding=1, quant_type=quant_type)  # 28x28 -> 14x14
        
        # Fully connected layers
        self.fc1 = BitLinear(64 * 14 * 14, 128, quant_type=quant_type)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv blocks
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="MNIST training with BitConv2d layers")
    parser.add_argument(
        "--quant-type",
        type=str,
        default="ai8pc_wpt",
        choices=["ai8pc_wpt", "ai8pg128_wpt", "ai8pg256_wpt"],
        help="Quantization type to use (default: ai8pc_wpt)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on test set after training"
    )
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Quantization type: {args.quant_type}")
    print(f"Hyperparameters: batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.num_epochs}")
    
    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    
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
    
    # Load test data if needed
    if args.test:
        test_dataset = datasets.MNIST(
            root='../../data',
            train=False,
            download=True,
            transform=transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Create model
    model = SimpleCNN(quant_type=args.quant_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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
            
            # Statistics
            running_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Print loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / num_batches
                accuracy = 100 * correct / total
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Print epoch summary
        avg_loss = running_loss / num_batches
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] completed - "
              f"Average Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
        print("-" * 60)
    
    print("\nTraining completed!")
    
    # Test evaluation
    if args.test:
        print("\nEvaluating on test set...")
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    main()


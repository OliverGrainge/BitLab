import bitlab.bnn as bnn
import torch.nn as nn 
import torch
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from torch import optim 
import torch.nn.functional as F


class MLP(bnn.Module): 
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int = 5):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(in_features, hidden_features))
        self.layers.append(nn.LayerNorm(hidden_features))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(bnn.BitLinear(hidden_features, hidden_features))
            self.layers.append(nn.LayerNorm(hidden_features))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def load_mnist_data(batch_size=128, num_workers=2):
    """Load MNIST dataset with appropriate transforms."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # Flatten to 784 features
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten to 784 features
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """Train the model for specified epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
    
    # Evaluate in eval mode
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy (eval mode): {test_acc:.2f}%")


def main():
    """Main training function."""
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 3
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=BATCH_SIZE)
    
    # Create model
    model = MLP(
        in_features=784,  # 28x28 flattened
        hidden_features=HIDDEN_SIZE,
        out_features=10,  # 10 classes
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Train model
    train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, device)
    
    # Deploy model (quantize weights)
    model = model.deploy()
    # Test deployed model
    _, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test Accuracy (deploy mode): {test_acc:.2f}%")


if __name__ == "__main__":
    main()
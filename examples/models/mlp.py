import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

from bitmodels import AutoBitModel
from bitmodels.mlp import BitMLPConfig, BitMLPModel

def detect_device():
    """Detect and return the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        device_info = f"CUDA ({device_name})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "Apple Metal Performance Shaders (MPS)"
    else:
        device = torch.device("cpu")
        device_info = "CPU"
    
    return device, device_info

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.view(-1)),  # flatten 28x28 to 784
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, optimizer, dataloader, device, loss_fn, progress, task):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = outputs.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        
        # Update progress
        progress.update(task, advance=1)
    
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * x.size(0)
            pred = outputs.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_results_table(results):
    """Create a nice table for displaying results."""
    table = Table(title="Training Results", box=box.ROUNDED)
    table.add_column("Epoch", style="cyan", no_wrap=True)
    table.add_column("Train Loss", style="green")
    table.add_column("Train Acc", style="green")
    table.add_column("Test Loss", style="blue")
    table.add_column("Test Acc", style="blue")
    table.add_column("Time (s)", style="magenta")
    
    for epoch, result in results.items():
        table.add_row(
            str(epoch),
            f"{result['train_loss']:.4f}",
            f"{result['train_acc']:.4f}",
            f"{result['test_loss']:.4f}",
            f"{result['test_acc']:.4f}",
            f"{result['time']:.2f}"
        )
    
    return table

def main():
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]BitLab MLP Training Example[/bold blue]\n"
        "Training a BitLinear MLP on MNIST dataset",
        border_style="blue"
    ))
    
    # Config for MNIST (use in_channels=784 for flattened input)
    config = BitMLPConfig(
        n_layers=5,
        in_channels=784,
        hidden_dim=128,
        out_channels=10,
        dropout=0.1,
    )

    # Use AutoBitModel for maximum flexibility
    model = AutoBitModel.from_config(config)

    # Detect and use the best available accelerator
    device, device_info = detect_device()
    model.to(device)

    # Display model info
    param_count = count_parameters(model)
    console.print(f"\n[bold]Model Information:[/bold]")
    console.print(f"  Type: {type(model).__name__}")
    console.print(f"  Parameters: {param_count:,}")
    console.print(f"  Device: {device_info}")
    console.print(f"  Architecture: {config.n_layers} layers, {config.hidden_dim} hidden units")

    # Load data
    console.print(f"\n[bold]Loading MNIST dataset...[/bold]")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    console.print(f"  Training samples: {len(train_loader.dataset):,}")
    console.print(f"  Test samples: {len(test_loader.dataset):,}")
    console.print(f"  Batches per epoch: {len(train_loader)}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop with progress tracking
    results = {}
    console.print(f"\n[bold]Starting Training...[/bold]")
    
    for epoch in range(1, 6):
        epoch_start_time = time.time()
        
        # Training phase with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            train_task = progress.add_task(f"Epoch {epoch} - Training", total=len(train_loader))
            train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, loss_fn, progress, train_task)
        
        # Evaluation phase
        eval_start = time.time()
        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
        eval_time = time.time() - eval_start
        
        epoch_time = time.time() - epoch_start_time
        
        # Store results
        results[epoch] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'time': epoch_time
        }
        
        # Display epoch results
        console.print(f"\n[bold cyan]Epoch {epoch} Results:[/bold cyan]")
        console.print(f"  [green]Train:[/green] Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        console.print(f"  [blue]Test:[/blue]  Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        console.print(f"  [magenta]Time:[/magenta] {epoch_time:.2f}s (eval: {eval_time:.2f}s)")

    # Display final results table
    console.print("\n")
    console.print(create_results_table(results))
    
    # Final summary
    best_test_acc = max(results[e]['test_acc'] for e in results)
    final_test_acc = results[max(results.keys())]['test_acc']
    
    console.print(f"\n[bold green]Training Complete![/bold green]")
    console.print(f"  Best test accuracy: {best_test_acc:.4f}")
    console.print(f"  Final test accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main()

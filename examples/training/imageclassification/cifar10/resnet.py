"""CIFAR-10 ResNet18 training with BitConv2d using BitImageClassifierTrainer."""

import sys
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from bitlab.bnn import Module, BitConv2d
from bitlab.bittrainer.classification import BitImageClassifierTrainer


class BasicBlock(nn.Module):
    """Basic ResNet block with parameterized convolution layer."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_layer=nn.Conv2d):
        super().__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(Module):
    """ResNet18 with parameterized convolution layers for CIFAR-10."""

    def __init__(self, num_classes=10, conv_layer=nn.Conv2d):
        super().__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.conv_layer))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_conv_layer(layer_type, quant_type=None):
    """Get the convolution layer class based on layer type."""
    if layer_type == 'standard':
        return nn.Conv2d
    elif layer_type == 'bitconv':
        if quant_type is None:
            raise ValueError("quant_type must be specified for bitconv layers")
        return partial(BitConv2d, quant_type=quant_type)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def main(config_path: str):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    pl.seed_everything(config["seed"], workers=True)
    
    print(f"Layer type: {config['layer_type']}")
    print(f"Quantization type: {config['quant_type']}")
    print(f"Run name: {config['run_name']}")
    print(f"Config file: {config_path}\n")
    
    # Data augmentation and normalization for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 data
    train_dataset = datasets.CIFAR10(
        root=config["data_root"],
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=config["data_root"],
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    # Create model with appropriate layer type
    conv_layer = get_conv_layer(config["layer_type"], config["quant_type"])
    model = ResNet18(num_classes=config["num_classes"], conv_layer=conv_layer)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * config["num_epochs"]
    
    # Create BitImageClassifierTrainer
    lit_model = BitImageClassifierTrainer(
        model=model,
        num_classes=config["num_classes"],
        loss_type="cross_entropy",
        label_smoothing=0.0,
        learning_rate=config["learning_rate"],
        lr_warmup_steps=0,  # No warmup for CIFAR-10
        lr_scheduler=config["scheduler"],
        max_lr_steps=max_steps,
        optimizer="sgd",
        weight_decay=config["weight_decay"],
        sgd_momentum=config["momentum"],
        sgd_nesterov=True,
        top_k=[1, 5],
        compute_per_class_metrics=False,
        log_samples_every_n_epochs=10,
        num_samples_to_log=16,
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        name=config["run_name"],
        config=config
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best_model_{config['run_name']}" + "-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator=config["device"] if config["device"] != "auto" else "auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
        precision="32-true",
        deterministic=False,
        enable_progress_bar=True,
    )
    
    # Train the model
    print("Starting training...")
    print("=" * 70)
    trainer.fit(lit_model, train_loader, val_loader)
    
    print("=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resnet.py <config_path>")
        print("Example: python resnet.py config_ai8pc_wpt.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)

"""CIFAR-10 ResNet18 training with BitConv2d using BitImageClassifierTrainer."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from bitlab.bnn import Module, BitConv2d
from bitlab.bittrainer.classification import BitImageClassifierTrainer

from datamodule import CIFAR10ClassificationDataModule
import torch.multiprocessing as mp

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


def get_conv_layer(layer_type: str, quant_type: str = None):
    """Get the convolution layer class based on layer type."""
    if layer_type == 'standard':
        return nn.Conv2d
    elif layer_type == 'bitconv':
        if quant_type is None:
            raise ValueError("quant_type must be specified for bitconv layers")
        return partial(BitConv2d, quant_type=quant_type)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def main(config_path: str) -> None:
    """Main training function."""
    torch.set_float32_matmul_precision('medium')
    
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    pl.seed_everything(config["seed"], workers=True)
    
    # Set up output directories in the current working directory
    output_dir = Path.cwd()
    config_name = Path(config_path).stem
    checkpoint_dir = output_dir / "checkpoints" / config_name
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print(f"Layer type: {config['model']['layer_type']}")
    print(f"Quantization type: {config['model']['quant_type']}")
    print(f"Run name: {config['run_name']}")
    print(f"Config file: {config_path}\n")
    
    # Create datamodule
    datamodule = CIFAR10ClassificationDataModule(
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        pin_memory=False,
        use_augmentation=config["use_augmentation"],
        seed=config["seed"],
    )
    
    # Create model with appropriate layer type
    conv_layer = get_conv_layer(config["model"]["layer_type"], config["model"]["quant_type"])
    model = ResNet18(num_classes=config["num_classes"], conv_layer=conv_layer)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Calculate total training steps for scheduler
    # We need to setup the datamodule to get the number of batches
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataloader())
    max_steps = steps_per_epoch * config["max_epochs"]
    
    # Create BitImageClassifierTrainer
    lit_model = BitImageClassifierTrainer(
        model=model,
        num_classes=config["num_classes"],
        loss_type=config["loss_type"],
        label_smoothing=config["label_smoothing"],
        learning_rate=config["learning_rate"],
        lr_warmup_steps=config["lr_warmup_steps"],
        lr_scheduler=config["lr_scheduler"],
        max_lr_steps=max_steps,
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
        sgd_momentum=config.get("sgd_momentum", 0.9),
        sgd_nesterov=config.get("sgd_nesterov", True),
        top_k=config.get("top_k", [1, 5]),
        compute_per_class_metrics=config.get("compute_per_class_metrics", False),
        log_samples_every_n_epochs=config.get("log_samples_every_n_epochs", 10),
        num_samples_to_log=config.get("num_samples_to_log", 16),
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=config_name,
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup wandb logger
    logger = WandbLogger(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=config["run_name"],
        save_dir=str(log_dir),
        config=config,
    )
    
    # Trainer kwargs
    trainer_kwargs = dict(
        default_root_dir=str(output_dir),
        max_epochs=config["max_epochs"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        gradient_clip_val=config.get("grad_clip_val", 0.0),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=False,
        enable_progress_bar=True,
    )
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train the model
    print("Starting training...")
    print("=" * 70)
    trainer.fit(lit_model, datamodule=datamodule)
    
    print("=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    if len(sys.argv) < 2:
        print("Error: Config file required!")
        print("Usage: python main.py <config.yaml>")
        sys.exit(1)
    
    main(sys.argv[1])



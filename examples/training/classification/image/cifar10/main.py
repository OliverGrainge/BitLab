"""CIFAR-10 ResNet18 training with BitConv2d using BitImageClassifierTrainer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from datamodule import CIFAR10ClassificationDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from bitlab.bitmodels.imageclassification import BitResNetConfig, BitResNetModel
from bitlab.bittrainer.callbacks import (ClassificationVisualizationLogger,
                                         GradientNormLogger,
                                         WeightHistogramLogger)
from bitlab.bittrainer.classification import BitImageClassificationTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str) -> None:
    """Main training function."""
    torch.set_float32_matmul_precision("medium")

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

    # Build model via BitResNetConfig/Model
    model_config = BitResNetConfig(
        num_classes=config["num_classes"],
        in_channels=config["model"].get("in_channels", 3),
        base_channels=config["model"].get("base_channels", 64),
        block_layers=tuple(config["model"].get("block_layers", [2, 2, 2, 2])),
        quant_type=config["model"].get("quant_type"),
    )
    model = BitResNetModel(model_config)

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
    lit_model = BitImageClassificationTrainer(
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

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logging_callbacks = [
        GradientNormLogger(
            every_n_steps=config.get("grad_norm_log_every_n_steps", 100)
        ),
        WeightHistogramLogger(
            log_every_n_epochs=config.get("weight_histogram_log_every_n_epochs", 1)
        ),
        ClassificationVisualizationLogger(
            num_samples_to_log=config.get("num_samples_to_log", 16),
            log_samples_every_n_epochs=config.get("log_samples_every_n_epochs", 10),
            log_confusion_matrix=config.get("log_confusion_matrix", True),
        ),
    ]

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
        callbacks=[checkpoint_callback, lr_monitor, *logging_callbacks],
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
    if len(sys.argv) < 2:
        print("Error: Config file required!")
        print("Usage: python main.py <config.yaml>")
        sys.exit(1)

    main(sys.argv[1])

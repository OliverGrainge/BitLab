"""MNIST MLP training with BitLinear layers using BitImageClassifierTrainer."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from bitlab.bnn import Module, BitLinear
from bitlab.bittrainer.classification import BitImageClassifierTrainer

from datamodule import MNISTClassificationDataModule


class MNISTMLP(Module):
    """Simple fully-connected network for MNIST classification."""

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_dims: Iterable[int] | None = None,
        num_classes: int = 10,
        linear_layer: type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims) if hidden_dims is not None else [256, 256]
        if not hidden_dims:
            hidden_dims = [256]

        layer_dims: List[int] = [input_size] + hidden_dims + [num_classes]

        layers: List[nn.Module] = []
        prev_dim = layer_dims[0]

        for idx, next_dim in enumerate(layer_dims[1:]):
            is_last = idx == len(layer_dims[1:]) - 1

            if idx == 0:
                layer_cls = nn.Linear
            elif is_last:
                layer_cls = nn.Linear
            else:
                layer_cls = linear_layer

            layers.append(layer_cls(prev_dim, next_dim))
            if not is_last:
                layers.append(nn.ReLU(inplace=True))

            prev_dim = next_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = x.view(x.size(0), -1)
        return self.net(x)


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_linear_layer(layer_type: str, quant_type: str | None = None):
    """Return the linear layer class based on the configured layer type."""

    if layer_type == "standard":
        return nn.Linear
    if layer_type == "bitlinear":
        if quant_type is None:
            raise ValueError("quant_type must be specified for bitlinear layers")
        return partial(BitLinear, quant_type=quant_type)
    raise ValueError(f"Unknown layer type: {layer_type}")


def main(config_path: str) -> None:
    """Main training entrypoint."""

    torch.set_float32_matmul_precision("medium")

    # Load configuration
    config = load_config(config_path)

    # Seed everything
    pl.seed_everything(config["seed"], workers=True)

    # Output directories relative to current working directory
    output_dir = Path.cwd()
    config_name = Path(config_path).stem
    checkpoint_dir = output_dir / "checkpoints" / config_name
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    print(f"Layer type: {config['model']['layer_type']}")
    print(f"Quantization type: {config['model'].get('quant_type')}")
    print(f"Run name: {config['run_name']}")
    print(f"Config file: {config_path}\n")

    # Create datamodule
    datamodule = MNISTClassificationDataModule(
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        use_augmentation=config.get("use_augmentation", False),
        seed=config["seed"],
    )

    # Build model
    linear_layer = get_linear_layer(
        config["model"]["layer_type"],
        config["model"].get("quant_type"),
    )
    model = MNISTMLP(
        hidden_dims=config["model"].get("hidden_dims", [256, 256]),
        num_classes=config["num_classes"],
        linear_layer=linear_layer,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Calculate total training steps
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataloader())
    max_steps = steps_per_epoch * config["max_epochs"]

    # Lightning module wrapper
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
        sgd_nesterov=config.get("sgd_nesterov", False),
        top_k=config.get("top_k", [1]),
        compute_per_class_metrics=config.get("compute_per_class_metrics", False),
        log_samples_every_n_epochs=config.get("log_samples_every_n_epochs", 5),
        num_samples_to_log=config.get("num_samples_to_log", 16),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"best_model_{config['run_name']}" + "-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = WandbLogger(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=config["run_name"],
        save_dir=str(log_dir),
        config=config,
    )

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

    trainer = pl.Trainer(**trainer_kwargs)

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


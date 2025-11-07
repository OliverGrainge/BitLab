"""Example training script for BitUNet with the BitDDPMTrainer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from bitlab.bitmodels import BitUNetConfig, BitUNetModel
from bitlab.bittrainer import BitDDPMTrainer

from datamodule import CIFAR10DataModule


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str) -> None:
    torch.set_float32_matmul_precision('medium')
    
    # Load configuration
    config = load_config(config_path)

    pl.seed_everything(config["seed"], workers=True)

    # Set up output directories in the current working directory
    output_dir = Path.cwd()
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    datamodule = CIFAR10DataModule(
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        val_split=config["val_split"],
        seed=config["seed"],
    )

    model_config = BitUNetConfig(
        image_size=config["model"]["image_size"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        model_channels=config["model"]["model_channels"],
        attention_resolutions=tuple(config["model"]["attention_resolutions"]),
        num_heads=config["model"]["num_heads"],
        num_res_blocks=config["model"]["num_res_blocks"],
        channel_mult=tuple(config["model"]["channel_mult"]),
    )
    
    model = BitUNetModel(model_config)

    diffusion_module = BitDDPMTrainer(
        model=model,
        image_size=model_config.image_size,
        in_channels=model_config.in_channels,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        optimizer=config["optimizer"],
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        loss_type=config["loss_type"],
        prediction_type=config["prediction_type"],
        use_ema=config["use_ema"],
        num_sample_steps=config["num_sample_steps"],
        sample_every_n_steps=config["sample_every_n_steps"],
        num_samples=config["num_samples"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="bitddpm-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = WandbLogger(
        project="bitddpm_unet",
        name="bitddpm_unet",
        save_dir=str(log_dir),
    )

    trainer_kwargs = dict(
        default_root_dir=str(output_dir),
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        max_steps=config["max_steps"],
        gradient_clip_val=config["grad_clip_val"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        log_every_n_steps=config["log_every_n_steps"],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=None,
    )

    if config["validate_every_n_steps"] > 0:
        trainer_kwargs["val_check_interval"] = config["validate_every_n_steps"]
    else:
        trainer_kwargs["limit_val_batches"] = 0

    trainer = pl.Trainer(**trainer_kwargs, check_val_every_n_epoch=None)

    trainer.fit(diffusion_module, datamodule=datamodule)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Config file required!")
        print("Usage: python main.py <config.yaml>")
        sys.exit(1)
    
    main(sys.argv[1])
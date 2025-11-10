"""Example training script for BitUNet with the BitDDPMTrainer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from bitlab.bitmodels import BitUNetConfig, BitUNetModel
from bitlab.bittrainer import BitDDPMTrainer
from bitlab.bittrainer.callbacks import (
    DiffusionSampleLogger,
    GradientNormLogger,
    WeightHistogramLogger,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
    config_name = Path(config_path).stem
    checkpoint_dir = output_dir / "checkpoints" / config_name
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    datamodule = CIFAR10DataModule(
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        val_split=config["val_split"],
        seed=config["seed"],
    )

    quant_type = config["model"].get("quant_type", BitUNetConfig.model_fields["quant_type"].default)
    print(f"Run name: {config['run_name']}")
    print(f"Quantization type: {quant_type}")
    print(f"Config file: {config_path}\n")

    model_config = BitUNetConfig(
        image_size=config["model"]["image_size"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        model_channels=config["model"]["model_channels"],
        attention_resolutions=tuple(config["model"]["attention_resolutions"]),
        num_heads=config["model"]["num_heads"],
        num_res_blocks=config["model"]["num_res_blocks"],
        channel_mult=tuple(config["model"]["channel_mult"]),
        dropout=config["model"]["dropout"],
        quant_type=quant_type,
    )
    
    model = BitUNetModel(model_config)

    diffusion_module = BitDDPMTrainer(
        model=model,
        image_size=model_config.image_size,
        in_channels=model_config.in_channels,
        # Diffusion parameters
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        # Loss configuration
        loss_type=config["loss_type"],
        prediction_type=config["prediction_type"],
        # Training parameters
        learning_rate=config["learning_rate"],
        lr_warmup_steps=config["lr_warmup_steps"],
        lr_scheduler=config["lr_scheduler"],
        max_lr_steps=config["max_lr_steps"],
        use_ema=config["use_ema"],
        ema_decay=config["ema_decay"],
        # Sampling parameters
        num_sample_steps=config["num_sample_steps"],
        sample_every_n_steps=config["sample_every_n_steps"],
        num_samples=config["num_samples"],
        # Optimizer parameters
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta1"],
        adam_beta2=config["adam_beta2"],
        adam_epsilon=config["adam_epsilon"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=config_name,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logging_callbacks = [
        GradientNormLogger(every_n_steps=config.get("grad_norm_log_every_n_steps", 100)),
        WeightHistogramLogger(log_every_n_epochs=config.get("weight_histogram_log_every_n_epochs", 1)),
        DiffusionSampleLogger(
            batch_size=config["num_samples"],
            num_steps=config["num_sample_steps"],
            log_every_n_epochs=config.get("sample_log_every_n_epochs", 1),
            use_ema=config.get("use_ema"),
        ),
    ]

    logger = WandbLogger(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=config["run_name"],
        save_dir=str(log_dir),
        config=config,
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
        callbacks=[checkpoint_callback, lr_monitor, *logging_callbacks],
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
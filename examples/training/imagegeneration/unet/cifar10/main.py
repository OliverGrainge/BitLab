"""Example training script for BitUNet with the BitDDPMTrainer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from bitlab.bitmodels.imagegeneration import BitUNetConfig, BitUNetModel
from bitlab.bittrainer.callbacks import (
    FIDAndInceptionCallback,
    GradientNormLogger,
    ImageSampleCallback,
    WeightHistogramLogger,
)
from bitlab.bittrainer.imagegeneration import BitImageGenerationTrainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datamodule import CIFAR10DataModule


def load_cfg(cfg_path: str) -> dict:
    """Load cfguration from a YAML file."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main(cfg_path: str) -> None:
    torch.set_float32_matmul_precision("medium")

    # Load cfguration
    cfg = load_cfg(cfg_path)

    pl.seed_everything(cfg["seed"], workers=True)

    # Set up output directories in the current working directory
    output_dir = Path.cwd()
    cfg_name = Path(cfg_path).stem
    checkpoint_dir = output_dir / "checkpoints" / cfg_name
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    datamodule = CIFAR10DataModule(
        train_batch_size=cfg["data"]["train_batch_size"],
        val_batch_size=cfg["data"]["val_batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    quant_type = cfg["model"].get(
        "quant_type", BitUNetConfig.model_fields["quant_type"].default
    )

    print(f"Run name: {cfg['experiment']['run_name']}")
    print(f"Quantization type: {quant_type}")
    print(f"cfg file: {cfg_path}\n")

    model_cfg = BitUNetConfig(
        image_size=cfg["model"]["image_size"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        model_channels=cfg["model"]["model_channels"],
        attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
        num_heads=cfg["model"]["num_heads"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        channel_mult=tuple(cfg["model"]["channel_mult"]),
        dropout=cfg["model"]["dropout"],
        quant_type=quant_type,
    )

    model = BitUNetModel(
        model_cfg,
        image_size=model_cfg.image_size,
        in_channels=model_cfg.in_channels,
        num_timesteps=cfg["diffusion"]["num_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        prediction_type=cfg["diffusion"]["prediction_type"],
        default_num_steps=cfg["sampling"]["num_sample_steps"],
    )

    diffusion_module = BitImageGenerationTrainer(
        model=model,
        loss_type=cfg["diffusion"]["loss_type"],
        learning_rate=float(cfg["optimizer"]["lr"]),
        lr_warmup_steps=cfg["scheduler"]["warmup_steps"],
        lr_scheduler=cfg["scheduler"]["name"],
        max_lr_steps=cfg["scheduler"]["max_steps"],
        use_ema=cfg["ema"]["enabled"],
        ema_decay=cfg["ema"]["decay"],
        num_sample_steps=cfg["sampling"]["num_sample_steps"],
        num_samples=cfg["sampling"]["num_samples"],
        optimizer=cfg["optimizer"]["name"],
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        adam_beta1=float(cfg["optimizer"]["beta1"]),
        adam_beta2=float(cfg["optimizer"]["beta2"]),
        adam_epsilon=float(cfg["optimizer"]["eps"]),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=cfg_name,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        GradientNormLogger(every_n_steps=500),
        WeightHistogramLogger(log_every_n_steps=5000),
        ImageSampleCallback(
            num_images=cfg["logging"]["images"]["num_images"],
            every_n_steps=cfg["logging"]["images"]["every_n_steps"],
            nrow=cfg["logging"]["images"]["nrow"],
            log_key=cfg["logging"]["images"]["log_key"],
            use_ema=cfg["logging"]["images"]["use_ema"],
        ),
    ]

    logger = WandbLogger(
        project=cfg["experiment"]["wandb_project"],
        name=cfg["experiment"]["run_name"],
        save_dir=str(log_dir),
    )

    trainer_kwargs = dict(
        default_root_dir=str(output_dir),
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_steps=cfg["trainer"]["max_steps"],
        gradient_clip_val=cfg["trainer"]["grad_clip_val"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        callbacks=[checkpoint_callback, lr_monitor, *callbacks],
        logger=logger,
        max_epochs=None,
    )

    if cfg["trainer"]["validate_every_n_steps"] > 0:
        trainer_kwargs["val_check_interval"] = cfg["trainer"]["validate_every_n_steps"]
    else:
        trainer_kwargs["limit_val_batches"] = 0

    trainer = pl.Trainer(**trainer_kwargs, check_val_every_n_epoch=None)

    trainer.fit(diffusion_module, datamodule=datamodule)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: cfg file required!")
        print("Usage: python main.py <cfg.yaml>")
        sys.exit(1)

    main(sys.argv[1])

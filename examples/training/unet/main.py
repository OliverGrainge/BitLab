"""Example training script for BitUNet with the BitDDPMTrainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from bitlab.bitmodels import BitUNetConfig, BitUNetModel
from bitlab.bittrainer import BitDDPMTrainer

from datamodule import LSUNBedroomsDataModule


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BitUNet diffusion model on LSUN Bedrooms")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bitddpm"), help="Directory for checkpoints and logs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Limit on training steps (-1 to disable)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator setting (e.g. 'gpu', 'cpu', 'auto')")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use (e.g. '1', 'auto', '0,1')")
    parser.add_argument("--precision", type=str, default="32-true", help="Precision setting for Lightning trainer")
    parser.add_argument("--train-batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--val-split", type=float, default=0.02, help="Fraction of training data to reserve for validation if none provided")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adamw", help="Optimizer type for BitDDPMTrainer")
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "cosine", "quadratic"], default="linear", help="Noise schedule type")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--loss-type", type=str, choices=["l1", "l2", "huber"], default="l2", help="Loss type for diffusion training")
    parser.add_argument("--prediction-type", type=str, choices=["epsilon", "x0", "v"], default="epsilon", help="Prediction type for the diffusion model")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA tracking (defaults to enabled)")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="Disable EMA tracking")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--num-sample-steps", type=int, default=50, help="Number of DDIM sampling steps during validation")
    parser.add_argument("--sample-every-n-steps", type=int, default=1000, help="How often (in steps) to generate samples during validation")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to log when sampling")
    parser.add_argument("--grad-clip-val", type=float, default=0.0, help="Gradient clipping value (0 disables clipping)")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Logging frequency in steps")
    parser.add_argument(
        "--validate-every-n-steps",
        type=int,
        default=1000,
        help="Run validation every N training steps (set to 0 to disable automatic validation)"
    )
    return parser.parse_args(cli_args)


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)

    pl.seed_everything(args.seed, workers=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = LSUNBedroomsDataModule(
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model_config = BitUNetConfig(
        image_size=64,
        in_channels=3,
        out_channels=3,
        model_channels=96,
        attention_resolutions=(2, 4),
        num_heads=2,
    )
    
    model = BitUNetModel(model_config)

    diffusion_module = BitDDPMTrainer(
        model=model,
        image_size=model_config.image_size,
        in_channels=model_config.in_channels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        use_ema=args.use_ema,
        num_sample_steps=args.num_sample_steps,
        sample_every_n_steps=args.sample_every_n_steps,
        num_samples=args.num_samples,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="bitddpm-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"),
        name="bitddpm_unet",
        default_hp_metric=False,
    )

    trainer_kwargs = dict(
        default_root_dir=str(output_dir),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=None,
    )

    if args.validate_every_n_steps > 0:
        trainer_kwargs["val_check_interval"] = args.validate_every_n_steps
    else:
        trainer_kwargs["limit_val_batches"] = 0

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(diffusion_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
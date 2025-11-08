"""
Sample from a trained BitDDPM model and compute the FID score on CIFAR-10.

Example
-------
python test.py config.yaml --checkpoint /path/to/checkpoint.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torchmetrics.image.fid import FrechetInceptionDistance

from bitlab.bitmodels import BitUNetConfig, BitUNetModel
from bitlab.bittrainer import BitDDPMTrainer

from datamodule import CIFAR10DataModule


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as fp:
        return yaml.safe_load(fp)


def build_diffusion_module(config: dict) -> Tuple[BitDDPMTrainer, BitUNetConfig]:
    model_cfg = BitUNetConfig(
        image_size=config["model"]["image_size"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        model_channels=config["model"]["model_channels"],
        attention_resolutions=tuple(config["model"]["attention_resolutions"]),
        num_heads=config["model"]["num_heads"],
        num_res_blocks=config["model"]["num_res_blocks"],
        channel_mult=tuple(config["model"]["channel_mult"]),
    )

    model = BitUNetModel(model_cfg)

    diffusion_module = BitDDPMTrainer(
        model=model,
        image_size=model_cfg.image_size,
        in_channels=model_cfg.in_channels,
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

    return diffusion_module, model_cfg


def load_checkpoint(module: BitDDPMTrainer, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at '{checkpoint_path}' does not contain a 'state_dict' entry."
        )

    module.load_state_dict(checkpoint["state_dict"])


def normalise_to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    # Inputs expected in [-1, 1]; convert to [0, 1]
    images = (images + 1.0) / 2.0
    return torch.clamp(images, 0.0, 1.0)


def update_fid_with_real_images(
    fid: FrechetInceptionDistance,
    datamodule: CIFAR10DataModule,
    num_samples: int,
    device: torch.device,
) -> int:
    datamodule.prepare_data()
    datamodule.setup(stage="val")

    dataloader = datamodule.val_dataloader()
    collected = 0

    for batch in dataloader:
        remaining = num_samples - collected
        if remaining <= 0:
            break

        images = batch[:remaining].to(device)
        images = normalise_to_unit_interval(images)
        fid.update(images, real=True)
        collected += images.shape[0]

    return collected


def update_fid_with_generated_images(
    fid: FrechetInceptionDistance,
    module: BitDDPMTrainer,
    num_samples: int,
    batch_size: int,
    num_steps: int | None,
    use_ema: bool | None,
) -> None:
    module.eval()

    generated = 0
    while generated < num_samples:
        current_batch = min(batch_size, num_samples - generated)
        with torch.no_grad():
            samples = module.sample_ddim(
                batch_size=current_batch,
                num_steps=num_steps,
                use_ema=use_ema,
            )

        samples = normalise_to_unit_interval(samples)
        fid.update(samples, real=False)
        generated += samples.shape[0]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for BitDDPM CIFAR-10 model.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration used for training.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PyTorch Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: automatically chosen).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to use when estimating FID.",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=256,
        help="Batch size for diffusion sampling.",
    )
    parser.add_argument(
        "--fid-feature",
        type=int,
        default=2048,
        help="Inception feature dimensionality for FID.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Override the number of DDIM sampling steps (defaults to config).",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA weights during sampling.",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    torch.set_float32_matmul_precision("medium")
    device = torch.device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

    datamodule = CIFAR10DataModule(
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        val_split=config["val_split"],
        seed=config["seed"],
    )

    diffusion_module, _ = build_diffusion_module(config)
    diffusion_module.to(device)

    load_checkpoint(diffusion_module, args.checkpoint, device=device)

    fid = FrechetInceptionDistance(feature=args.fid_feature, normalize=True).to(device)

    target_samples = args.num_samples
    real_samples = update_fid_with_real_images(fid, datamodule, target_samples, device)
    if real_samples < target_samples:
        print(
            f"Warning: Requested {target_samples} real samples but only gathered {real_samples}."
        )
        target_samples = real_samples

    update_fid_with_generated_images(
        fid,
        diffusion_module,
        target_samples,
        args.gen_batch_size,
        args.sample_steps,
        None if not args.no_ema else False,
    )

    fid_score = fid.compute().item()
    print(f"FID score ({target_samples} samples): {fid_score:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


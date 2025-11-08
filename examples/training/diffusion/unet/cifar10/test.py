"""
Sample from a trained BitDDPM model and compute the FID score on CIFAR-10.

Example
-------
python test.py config.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml
try:
    from cleanfid import fid as cleanfid
except ImportError as exc:
    raise ImportError(
        "CleanFID is required. Install it with `pip install cleanfid`."
    ) from exc

from PIL import Image
from tqdm.auto import tqdm

from bitlab.bitmodels import BitUNetConfig, BitUNetModel
from bitlab.bittrainer import BitDDPMTrainer

from datamodule import CIFAR10DataModule



def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

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

    ema_state = checkpoint.get("ema_state_dict")
    if ema_state is not None and module.use_ema and module.ema_model is not None:
        module.ema_model.load_state_dict(ema_state)
    elif module.use_ema and module.ema_model is not None:
        print(
            "Warning: EMA state missing from checkpoint; using non-EMA weights for sampling."
        )


def normalise_to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    # Inputs expected in [-1, 1]; convert to [0, 1]
    images = (images + 1.0) / 2.0
    return torch.clamp(images, 0.0, 1.0)


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = normalise_to_unit_interval(images)
    images = images.mul(255.0).add(0.5).clamp(0.0, 255.0)
    return images.to(torch.uint8)


def save_images_to_directory(
    images: torch.Tensor,
    destination: Path,
    start_index: int,
) -> tuple[int, int, Path | None]:
    uint8_images = to_uint8(images.detach().cpu()).permute(0, 2, 3, 1).contiguous()
    arrays = uint8_images.numpy()

    first_path: Path | None = None

    for offset, array in enumerate(arrays):
        image = Image.fromarray(array)
        path = destination / f"{start_index + offset:06d}.png"
        image.save(path)
        if first_path is None:
            first_path = path

    saved = arrays.shape[0]
    return start_index + saved, saved, first_path


def export_real_images(
    datamodule: CIFAR10DataModule,
    num_samples: int,
    destination: Path,
) -> int:
    datamodule.prepare_data()
    datamodule.setup(stage="val")

    dataloader = datamodule.val_dataloader()
    collected = 0

    with tqdm(total=num_samples, desc="Collecting real images", unit="img") as progress:
        for batch in dataloader:
            remaining = num_samples - collected
            if remaining <= 0:
                break

            images = batch[:remaining]
            collected, saved, _ = save_images_to_directory(images, destination, collected)
            progress.update(saved)

    return collected


def export_generated_images(
    module: BitDDPMTrainer,
    num_samples: int,
    batch_size: int,
    num_steps: int | None,
    use_ema: bool | None,
    destination: Path,
    preview_path: Path | None = None,
) -> Path | None:
    module.eval()

    generated = 0
    first_saved: Path | None = None
    with tqdm(total=num_samples, desc="Generating samples", unit="img") as progress:
        while generated < num_samples:
            current_batch = min(batch_size, num_samples - generated)
            with torch.no_grad():
                samples = module.sample_ddim(
                    batch_size=current_batch,
                    num_steps=num_steps,
                    use_ema=use_ema,
                )

            generated, saved, batch_first = save_images_to_directory(
                samples, destination, generated
            )
            if first_saved is None and batch_first is not None:
                first_saved = batch_first
            progress.update(saved)

    if preview_path is not None and first_saved is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(first_saved, preview_path)

    return first_saved


def infer_checkpoint_path(config_path: Path) -> Path:
    config_name = config_path.stem
    candidates = [
        Path.cwd(),
        config_path.resolve().parent,
        Path(__file__).resolve().parent,
    ]
    for base in candidates:
        candidate = base / "checkpoints" / config_name / f"{config_name}.ckpt"
        if candidate.exists():
            return candidate
    return Path.cwd() / "checkpoints" / config_name / f"{config_name}.ckpt"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for BitDDPM CIFAR-10 model.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration used for training.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the PyTorch Lightning checkpoint (.ckpt). Defaults to checkpoints/<config>/<config>.ckpt.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=24,
        help="Number of samples to use when estimating FID.",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=8,
        help="Batch size for diffusion sampling.",
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
    parser.add_argument(
        "--cleanfid-batch-size",
        type=int,
        default=256,
        help="Batch size used by CleanFID when extracting features.",
    )
    parser.add_argument(
        "--cleanfid-workers",
        type=int,
        default=0,
        help="Number of worker processes CleanFID should use when reading images.",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    torch.set_float32_matmul_precision("medium")
    device = get_device()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    checkpoint_path = args.checkpoint or infer_checkpoint_path(args.config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

    load_checkpoint(diffusion_module, checkpoint_path, device=device)

    target_samples = args.num_samples

    with tempfile.TemporaryDirectory() as real_dir_name, tempfile.TemporaryDirectory() as gen_dir_name:
        real_dir = Path(real_dir_name)
        gen_dir = Path(gen_dir_name)

        real_samples = export_real_images(datamodule, target_samples, real_dir)
        if real_samples < target_samples:
            print(
                f"Warning: Requested {target_samples} real samples but only gathered {real_samples}."
            )
            target_samples = real_samples

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        preview_sample_path = (
            Path(__file__).resolve().parent / "logs" / "images" / run_id / "sample-000000.png"
        )

        export_generated_images(
            diffusion_module,
            target_samples,
            args.gen_batch_size,
            args.sample_steps,
            None if not args.no_ema else False,
            gen_dir,
            preview_path=preview_sample_path,
        )

        fid_score = cleanfid.compute_fid(
            str(real_dir),
            str(gen_dir),
            batch_size=args.cleanfid_batch_size,
            device=str(device),
            num_workers=args.cleanfid_workers,
        )

    title = f"Results for {args.config}"
    separator = "-" * len(title)
    print(f"\n{title}\n{separator}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Samples evaluated: {target_samples}")
    print(f"FID score        : {fid_score:.4f}")
    if preview_sample_path.exists():
        print(f"Preview sample   : {preview_sample_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


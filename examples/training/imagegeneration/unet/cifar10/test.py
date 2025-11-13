"""Evaluation script for BitUNet diffusion model - computes FID score."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from bitlab.bitmodels.imagegeneration import BitUNetConfig, BitUNetModel
from bitlab.bittrainer.imagegeneration import BitImageGenerationTrainer
from torchvision.utils import save_image
from tqdm import tqdm

from datamodule import CIFAR10DataModule


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_checkpoint(config_path: Path) -> Path | None:
    """Find checkpoint file based on config name.
    
    Searches in: ./checkpoints/{config_name}/{config_name}.ckpt
    Falls back to first .ckpt file found in directory.
    """
    config_name = config_path.stem
    checkpoint_dir = Path.cwd() / "checkpoints" / config_name
    
    if not checkpoint_dir.exists():
        return None
    
    # Try exact match first
    checkpoint_path = checkpoint_dir / f"{config_name}.ckpt"
    if checkpoint_path.exists():
        return checkpoint_path
    
    # Fall back to any .ckpt file
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    return ckpt_files[0] if ckpt_files else None


def load_model(config: dict, checkpoint_path: Path) -> BitImageGenerationTrainer:
    """Load model from checkpoint with configuration."""
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
        quant_type=config["model"]["quant_type"],
    )

    model = BitUNetModel(
        model_config,
        image_size=model_config.image_size,
        in_channels=model_config.in_channels,
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        prediction_type=config["diffusion"]["prediction_type"],
        default_num_steps=config["sampling"]["num_sample_steps"],
    )
    
    return BitImageGenerationTrainer.load_from_checkpoint(
        checkpoint_path, model=model, map_location="cpu"
    )


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to [0, 1] range."""
    if images.min() < 0:
        images = (images + 1) / 2
    return torch.clamp(images, 0, 1)


def save_sample_images(images: torch.Tensor, save_dir: Path, prefix: str, start_idx: int) -> None:
    """Save individual images to disk."""
    for idx, img in enumerate(images.detach().cpu(), start=start_idx):
        save_image(img, save_dir / f"{prefix}_{idx:02d}.png")


def process_real_images(
    dataloader,
    fid_metric,
    num_samples: int,
    device: str,
    save_dir: Path | None = None,
    samples_to_save: int = 10,
) -> None:
    """Process real images and update FID metric."""
    num_processed = 0
    num_saved = 0
    
    for batch in tqdm(dataloader, desc="Processing real images"):
        if num_processed >= num_samples:
            break
        
        images = batch.to(device)
        images = normalize_images(images)
        
        # Limit to num_samples
        remaining = num_samples - num_processed
        if images.shape[0] > remaining:
            images = images[:remaining]
        
        # Save examples
        if save_dir and num_saved < samples_to_save:
            to_save = min(images.shape[0], samples_to_save - num_saved)
            save_sample_images(images[:to_save], save_dir, "real", num_saved + 1)
            num_saved += to_save
        
        # Update FID metric
        images_uint8 = (images * 255).to(torch.uint8)
        fid_metric.update(images_uint8, real=True)
        num_processed += images.shape[0]


def generate_fake_images(
    model,
    fid_metric,
    num_samples: int,
    batch_size: int,
    save_dir: Path | None = None,
    samples_to_save: int = 10,
) -> None:
    """Generate fake images and update FID metric."""
    num_generated = 0
    num_saved = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating fake images"):
            current_batch_size = min(batch_size, num_samples - num_generated)
            
            images = model.generate_samples(
                batch_size=current_batch_size,
                use_ema=True,
            )
            images = normalize_images(images)
            
            # Save examples
            if save_dir and num_saved < samples_to_save:
                to_save = min(images.shape[0], samples_to_save - num_saved)
                save_sample_images(images[:to_save], save_dir, "generated", num_saved + 1)
                num_saved += to_save
            
            # Update FID metric
            images_uint8 = (images * 255).to(torch.uint8)
            fid_metric.update(images_uint8, real=False)
            num_generated += current_batch_size


def compute_fid_score(
    model: BitImageGenerationTrainer,
    datamodule: CIFAR10DataModule,
    num_samples: int = 10000,
    batch_size: int = 100,
    device: str = "cuda",
    save_dir: Path | None = None,
    samples_to_save: int = 10,
) -> float:
    """Compute FID score between generated and real images."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("Error: torchmetrics is required for FID computation.")
        print("Install with: pip install torchmetrics")
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print("Computing FID Score")
    print(f"{'=' * 60}")
    print(f"Samples: {num_samples} | Batch size: {batch_size} | Device: {device}\n")
    
    # Initialize
    fid = FrechetInceptionDistance(normalize=True).to(device)
    model = model.to(device).eval()
    
    # Process real images
    datamodule.setup(stage="test")
    real_loader = datamodule.val_dataloader()
    process_real_images(real_loader, fid, num_samples, device, save_dir, samples_to_save)
    
    # Generate fake images
    generate_fake_images(model, fid, num_samples, batch_size, save_dir, samples_to_save)
    
    # Compute FID
    print("\nComputing FID score...")
    fid_score = float(fid.compute())
    
    print(f"\n{'=' * 60}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'=' * 60}\n")
    
    return fid_score


def save_results(config_path: Path, checkpoint_path: Path, fid_score: float) -> None:
    """Save evaluation results to file."""
    results_dir = config_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{config_path.stem}_fid.txt"
    
    with open(results_file, "w") as f:
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"FID Score: {fid_score:.2f}\n")
    
    print(f"Results saved to: {results_file}")


def main(config_path: str) -> None:
    """Main evaluation function."""
    torch.set_float32_matmul_precision("medium")
    
    # Load configuration
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    
    print(f"\n{'=' * 60}")
    print("BitUNet Diffusion Model Evaluation")
    print(f"{'=' * 60}")
    print(f"Config: {config_path.name}")
    print(f"Run: {config['experiment']['run_name']}")
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(config_path)
    if checkpoint_path is None:
        print(f"\nError: No checkpoint found for '{config_path.stem}'")
        print(f"Expected location: ./checkpoints/{config_path.stem}/")
        sys.exit(1)
    
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"{'=' * 60}\n")
    
    # Setup data
    datamodule = CIFAR10DataModule(
        train_batch_size=config["data"]["train_batch_size"],
        val_batch_size=config["data"]["val_batch_size"],
        num_workers=config["data"]["num_workers"],
        val_split=config["data"]["val_split"],
        seed=config["seed"],
    )
    datamodule.prepare_data()
    
    # Load model
    print("Loading model...")
    model = load_model(config, checkpoint_path)
    print("Model loaded!\n")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Using CPU (this will be slow)\n")
    
    # Compute FID
    samples_dir = config_path.parent / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    fid_score = compute_fid_score(
        model=model,
        datamodule=datamodule,
        num_samples=1000,
        batch_size=100,
        device=device,
        save_dir=samples_dir,
        samples_to_save=10,
    )
    
    # Save results
    save_results(config_path, checkpoint_path, fid_score)
    print(f"Sample images saved to: {samples_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <config.yaml>")
        sys.exit(1)
    
    main(sys.argv[1])
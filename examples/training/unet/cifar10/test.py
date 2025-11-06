"""Test script for computing FID scores on a trained diffusion model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# Import your trainer
from bitlab.bittrainer import BitDDPMTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID score for a trained diffusion model")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate for FID computation (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for generation (default: 64)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps (default: 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name for computing real statistics (default: cifar10)"
    )
    parser.add_argument(
        "--num-real-samples",
        type=int,
        default=10000,
        help="Number of real samples to use for FID computation (default: 10000)"
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights if available (default: True)"
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Don't use EMA weights"
    )
    parser.add_argument(
        "--save-samples",
        type=str,
        default=None,
        help="Optional path to save generated samples as .npz file"
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> BitDDPMTrainer:
    """Load the trained model from a Lightning checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    model = BitDDPMTrainer.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Image size: {model.image_size}")
    print(f"Channels: {model.in_channels}")
    print(f"Timesteps: {model.num_timesteps}")
    print(f"EMA enabled: {model.use_ema}")
    
    return model


@torch.no_grad()
def generate_samples(
    model: BitDDPMTrainer,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    device: str,
    use_ema: bool,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate samples from the diffusion model."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Using {'EMA' if use_ema and model.ema_model else 'standard'} weights")
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Generate samples
        samples = model.sample_ddim(
            batch_size=current_batch_size,
            num_steps=num_steps,
            use_ema=use_ema
        )
        
        # Convert to [0, 255] uint8
        samples = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        samples = torch.clamp(samples, 0.0, 1.0)
        samples = (samples * 255).cpu().numpy().astype(np.uint8)
        
        all_samples.append(samples)
    
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    print(f"Generated samples shape: {all_samples.shape}")
    
    return all_samples


def load_real_images(
    dataset_name: str,
    num_samples: int,
    image_size: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """Load real images from the dataset."""
    print(f"\nLoading {num_samples} real images from {dataset_name}...")
    
    # Load dataset
    if dataset_name == "cifar10":
        dataset = load_dataset("cifar10", split="test")
    else:
        dataset = load_dataset(dataset_name, split="test")
    
    # Shuffle with seed
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    
    # Take the required number of samples
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Extract images
    images = []
    for item in tqdm(dataset, desc="Loading real images"):
        img = item["img"]
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize if needed
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size))
        
        # Convert to numpy array [C, H, W] in [0, 255]
        img_array = np.array(img).transpose(2, 0, 1)
        images.append(img_array)
    
    images = np.stack(images, axis=0)
    print(f"Real images shape: {images.shape}")
    
    return images


def compute_fid_pytorch_fid(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    batch_size: int = 64,
    device: str = "cuda"
) -> float:
    """
    Compute FID score using pytorch-fid library.
    
    Args:
        real_images: Real images [N, C, H, W] in [0, 255] uint8
        fake_images: Generated images [N, C, H, W] in [0, 255] uint8
        batch_size: Batch size for computing features
        device: Device to use
    
    Returns:
        FID score
    """
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        raise ImportError(
            "pytorch-fid is required for FID computation. "
            "Install it with: pip install pytorch-fid"
        )
    
    print("\nComputing FID score using pytorch-fid...")
    
    # Initialize InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    def get_activations(images: np.ndarray) -> np.ndarray:
        """Extract InceptionV3 features for images."""
        # Convert to torch tensor [N, C, H, W] in [0, 1]
        images_tensor = torch.from_numpy(images).float() / 255.0
        
        all_activations = []
        num_batches = (len(images_tensor) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(images_tensor))
                batch = images_tensor[start_idx:end_idx].to(device)
                
                # Resize to 299x299 for InceptionV3
                if batch.shape[-1] != 299:
                    batch = torch.nn.functional.interpolate(
                        batch,
                        size=(299, 299),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Get features
                pred = inception_model(batch)[0]
                
                # Flatten if needed
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
                
                activations = pred.squeeze(3).squeeze(2).cpu().numpy()
                all_activations.append(activations)
        
        return np.concatenate(all_activations, axis=0)
    
    # Get activations
    print("Computing features for real images...")
    real_activations = get_activations(real_images)
    
    print("Computing features for generated images...")
    fake_activations = get_activations(fake_images)
    
    # Compute statistics
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)
    
    # Compute FID
    fid = fid_score.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid


def compute_fid_cleanfid(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    device: str = "cuda"
) -> float:
    """
    Compute FID score using clean-fid library (more accurate).
    
    Args:
        real_images: Real images [N, C, H, W] in [0, 255] uint8
        fake_images: Generated images [N, C, H, W] in [0, 255] uint8
        device: Device to use
    
    Returns:
        FID score
    """
    try:
        from cleanfid import fid as clean_fid
    except ImportError:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install it with: pip install clean-fid"
        )
    
    print("\nComputing FID score using clean-fid...")
    
    # clean-fid expects images as numpy arrays [N, H, W, C]
    real_images_hwc = real_images.transpose(0, 2, 3, 1)
    fake_images_hwc = fake_images.transpose(0, 2, 3, 1)
    
    # Compute FID
    fid = clean_fid.compute_fid(
        real_images_hwc,
        fake_images_hwc,
        mode="clean",
        device=torch.device(device),
        verbose=True
    )
    
    return fid


def main():
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint_path, args.device)
    
    # Generate samples
    fake_images = generate_samples(
        model=model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device=args.device,
        use_ema=args.use_ema,
        seed=args.seed
    )
    
    # Save samples if requested
    if args.save_samples:
        save_path = Path(args.save_samples)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, samples=fake_images)
        print(f"\nSaved generated samples to: {save_path}")
    
    # Load real images
    real_images = load_real_images(
        dataset_name=args.dataset,
        num_samples=args.num_real_samples,
        image_size=model.image_size,
        seed=args.seed
    )
    
    # Compute FID score
    # Try clean-fid first (more accurate), fall back to pytorch-fid
    try:
        fid = compute_fid_cleanfid(real_images, fake_images, args.device)
        method = "clean-fid"
    except ImportError:
        print("\nclean-fid not available, falling back to pytorch-fid")
        try:
            fid = compute_fid_pytorch_fid(real_images, fake_images, args.batch_size, args.device)
            method = "pytorch-fid"
        except ImportError:
            print("\nError: Neither clean-fid nor pytorch-fid is installed.")
            print("Please install one of them:")
            print("  pip install clean-fid")
            print("  pip install pytorch-fid")
            return
    
    # Print results
    print("\n" + "=" * 60)
    print(f"FID Score ({method}): {fid:.2f}")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint_path}")
    print(f"Generated samples: {args.num_samples}")
    print(f"Real samples: {args.num_real_samples}")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Used EMA: {args.use_ema and model.ema_model is not None}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
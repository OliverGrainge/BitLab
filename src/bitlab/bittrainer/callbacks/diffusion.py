from typing import Optional, Any, List
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import pytorch_fid.fid_score as fid_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class FIDCallback(Callback):
    """
    Optimized version that uses precomputed statistics.
    
    This version precomputes and caches the statistics of real images,
    avoiding redundant feature extraction on every FID computation.
    
    This is the most efficient approach for repeated FID computations.
    
    Args:
        val_dataloader: DataLoader with real validation data
        stats_path: Path to save/load real image statistics (.npz file)
        num_samples: Number of samples to generate (default: 5000)
        every_n_epochs: Compute FID every N epochs (default: 5)
        every_n_steps: Compute FID every N steps (default: None)
        num_sample_steps: DDIM steps for sampling (default: 50)
        use_ema: Use EMA model if available (default: True)
        sample_batch_size: Batch size for sample generation (default: 16)
        fid_batch_size: Batch size for FID computation (default: 50)
        cleanup_after: Delete temp directories after computation (default: True)
        dims: Dimensionality of Inception features (default: 2048)
        num_workers: Workers for FID computation (default: 4)
        max_real_samples: Maximum real samples for statistics (default: 10000)
        save_samples_dir: Optional directory to persist generated and real images for inspection
    """

    def __init__(
        self,
        val_dataloader: DataLoader,
        stats_path: str = "./real_images_stats.npz",
        num_samples: int = 5000,
        every_n_epochs: int = 5,
        every_n_steps: Optional[int] = None,
        num_sample_steps: int = 50,
        use_ema: bool = True,
        sample_batch_size: int = 16,
        fid_batch_size: int = 50,
        cleanup_after: bool = True,
        dims: int = 2048,
        num_workers: int = 4,
        max_real_samples: int = 10000,
        save_samples_dir: Optional[str] = None,
    ):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.stats_path = Path(stats_path)
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.num_sample_steps = num_sample_steps
        self.use_ema = use_ema
        self.sample_batch_size = sample_batch_size
        self.fid_batch_size = fid_batch_size
        self.cleanup_after = cleanup_after
        self.dims = dims
        self.num_workers = num_workers
        self.max_real_samples = max_real_samples
        self.save_samples_dir = Path(save_samples_dir).expanduser() if save_samples_dir else None
        
        self.gen_dir: Optional[Path] = None
        self._temp_base_dir: Optional[Path] = None
        self._real_stats_computed = False
        self.generated_images_dir: Optional[Path] = None
        self.real_images_dir: Optional[Path] = None
        self._real_images_cached = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Initialize and precompute real image statistics."""
        if stage == "fit":
            if self.save_samples_dir is None:
                self.save_samples_dir = Path.cwd() / "fid_samples"

            self._temp_base_dir = Path(tempfile.mkdtemp(prefix="fid_callback_stats_"))
            self.gen_dir = self._temp_base_dir / "generated_images"
            self.gen_dir.mkdir(parents=True, exist_ok=True)

            if self.save_samples_dir is not None:
                self.save_samples_dir.mkdir(parents=True, exist_ok=True)
                self.generated_images_dir = self.save_samples_dir / "generated"
                self.real_images_dir = self.save_samples_dir / "real"
                self.generated_images_dir.mkdir(parents=True, exist_ok=True)
                self.real_images_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"PyTorchFIDWithStatsCallback initialized")
            print(f"Generated images dir: {self.gen_dir}")
            print(f"Statistics path: {self.stats_path}")
            if self.save_samples_dir is not None:
                print(f"Persistent samples dir: {self.save_samples_dir}")
            print(f"{'='*60}\n")
            
            # Compute or load real image statistics
            if not self.stats_path.exists():
                print("Precomputing real image statistics...")
                self._compute_real_statistics(pl_module.device)
                print(f"✓ Statistics saved to: {self.stats_path}\n")
            else:
                print(f"✓ Using cached statistics from: {self.stats_path}\n")
                self._real_stats_computed = True
                # Ensure a copy of the real images exists for inspection when requested
                if self.real_images_dir is not None:
                    self._ensure_real_images_cached(pl_module.device)

    def _save_real_images_to_destinations(
        self,
        device: torch.device,
        destinations: List[Path],
        skip_existing: bool = False,
    ) -> int:
        """Save real images to the provided destination directories."""
        if not destinations:
            return 0

        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)

        img_idx = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                batch = batch.to(device)
                images = ((batch + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

                for img in images:
                    filename = f"real_{img_idx:05d}.png"
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    if img_np.shape[2] == 1:
                        img_np = img_np.squeeze(2)

                    img_pil = Image.fromarray(img_np)

                    for destination in destinations:
                        target = destination / filename
                        if skip_existing and target.exists():
                            continue
                        img_pil.save(target)

                    img_idx += 1

                    if img_idx >= self.max_real_samples:
                        break
                if img_idx >= self.max_real_samples:
                    break

        if self.real_images_dir is not None and self.real_images_dir in destinations and img_idx > 0:
            self._real_images_cached = True

        return img_idx

    def _ensure_real_images_cached(self, device: torch.device) -> None:
        """Ensure real validation images are saved for inspection."""
        if self.real_images_dir is None or self._real_images_cached:
            return

        existing = list(self.real_images_dir.glob("real_*.png"))
        if existing:
            self._real_images_cached = True
            return

        print("Caching real validation images for inspection...")
        count = self._save_real_images_to_destinations(
            device,
            destinations=[self.real_images_dir],
            skip_existing=False,
        )
        print(f"  Cached {count} real images for inspection")

    def _compute_real_statistics(self, device: torch.device) -> None:
        """Compute and save statistics of real images."""
        # Create temporary directory for real images
        real_dir = self._temp_base_dir / "real_images_temp"
        real_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save real images
            destinations = [real_dir]
            if self.real_images_dir is not None:
                destinations.append(self.real_images_dir)

            img_idx = self._save_real_images_to_destinations(
                device,
                destinations=destinations,
                skip_existing=False,
            )
            
            print(f"  Saved {img_idx} real images")
            
            # Compute statistics using pytorch-fid
            print("  Computing Inception statistics...")
            from pytorch_fid.inception import InceptionV3
            
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx]).to(device)
            
            real_image_paths = sorted(str(p) for p in real_dir.glob("*.png"))
            if not real_image_paths:
                raise RuntimeError("No real images were saved for FID computation.")
            m, s = fid_score.calculate_activation_statistics(
                real_image_paths,
                model,
                batch_size=self.fid_batch_size,
                dims=self.dims,
                device=device,
                num_workers=self.num_workers,
            )
            
            # Save statistics
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.stats_path, mu=m, sigma=s)
            
            self._real_stats_computed = True
            
        finally:
            # Cleanup temporary real images directory
            if real_dir.exists():
                shutil.rmtree(real_dir)

    def _generate_and_save_samples(self, pl_module: "BitImageDiffusionTrainer") -> None:
        """Generate samples and save to disk."""
        # Clear previous images
        for f in self.gen_dir.glob("*.png"):
            f.unlink()
        
        print(f"Generating {self.num_samples} samples...")
        
        num_generated = 0
        img_idx = 0
        
        while num_generated < self.num_samples:
            batch_size = min(self.sample_batch_size, self.num_samples - num_generated)
            
            samples = pl_module.sample_ddim(
                batch_size=batch_size,
                num_steps=self.num_sample_steps,
                use_ema=self.use_ema,
            )
            
            images = ((samples + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            
            for img in images:
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                if img_np.shape[2] == 1:
                    img_np = img_np.squeeze(2)
                
                img_pil = Image.fromarray(img_np)
                img_pil.save(self.gen_dir / f"gen_{img_idx:05d}.png")
                img_idx += 1
            
            num_generated += batch_size
            
            if num_generated % 500 == 0 or num_generated == self.num_samples:
                print(f"  Generated {num_generated}/{self.num_samples} samples")
    
    def _persist_generated_images(self, context: Optional[str]) -> Optional[Path]:
        """Persist generated images to the configured inspection directory."""
        if self.generated_images_dir is None:
            return None

        context_name = context or "latest"
        context_dir = self.generated_images_dir / context_name

        if context_dir.exists():
            shutil.rmtree(context_dir)
        context_dir.mkdir(parents=True, exist_ok=True)

        for img_file in self.gen_dir.glob("*.png"):
            shutil.copy2(img_file, context_dir / img_file.name)

        # Maintain a marker file to indicate the freshest context
        latest_marker = self.generated_images_dir / "LATEST.txt"
        latest_marker.write_text(context_name)

        return context_dir

    @torch.no_grad()
    def compute_fid(
        self,
        pl_module: "BitImageDiffusionTrainer",
        context: Optional[str] = None,
    ) -> float:
        """Compute FID using precomputed statistics."""
        if not self._real_stats_computed:
            raise RuntimeError("Real statistics not computed. Call setup() first.")
        
        # Generate samples
        self._generate_and_save_samples(pl_module)

        persisted_dir = self._persist_generated_images(context)
        
        # Load real statistics
        print("Loading precomputed real statistics...")
        stats = np.load(self.stats_path)
        m1, s1 = stats['mu'], stats['sigma']
        
        # Compute generated statistics
        print("Computing generated image statistics...")
        from pytorch_fid.inception import InceptionV3
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        model = InceptionV3([block_idx]).to(pl_module.device)
        
        generated_image_paths = sorted(str(p) for p in self.gen_dir.glob("*.png"))
        if not generated_image_paths:
            raise RuntimeError("No generated images were found for FID computation.")
        m2, s2 = fid_score.calculate_activation_statistics(
            generated_image_paths,
            model,
            batch_size=self.fid_batch_size,
            dims=self.dims,
            device=pl_module.device,
            num_workers=self.num_workers,
        )
        
        # Compute FID
        fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
        
        print(f"✓ FID computed: {fid_value:.3f}")
        if persisted_dir is not None:
            print(f"  Generated samples saved to: {persisted_dir}")
        return fid_value

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Compute FID at validation epoch end."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        
        if trainer.sanity_checking:
            return
        
        print(f"\n{'='*60}")
        print(f"Computing FID at epoch {trainer.current_epoch}")
        print(f"{'='*60}")
        
        try:
            context_name = f"epoch_{trainer.current_epoch:04d}"
            fid_value = self.compute_fid(pl_module, context=context_name)
            pl_module.log("val/fid", fid_value, sync_dist=True, prog_bar=True)
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ FID computation failed: {e}")
            import traceback
            traceback.print_exc()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Optionally compute FID every N training steps."""
        if self.every_n_steps is None:
            return
        
        if trainer.global_step % self.every_n_steps != 0 or trainer.global_step == 0:
            return
        
        print(f"\n{'='*60}")
        print(f"Computing FID at step {trainer.global_step}")
        print(f"{'='*60}")
        
        try:
            context_name = f"step_{trainer.global_step:08d}"
            if trainer.current_epoch is not None:
                context_name = f"epoch_{trainer.current_epoch:04d}_{context_name}"
            fid_value = self.compute_fid(pl_module, context=context_name)
            pl_module.log("train/fid", fid_value, sync_dist=True)
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ FID computation failed: {e}")

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Cleanup temporary directories."""
        if self.cleanup_after and self._temp_base_dir is not None:
            try:
                shutil.rmtree(self._temp_base_dir)
                print(f"✓ Cleaned up temporary directory")
            except Exception as e:
                print(f"⚠️  Warning: Could not cleanup temp directory: {e}")
from typing import Optional
import math
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
from contextlib import contextmanager




class ImageSampleCallback(pl.Callback):
    """
    Log a grid of generated images every N training steps.

    Assumes `pl_module.generate_samples(batch_size=..., use_ema=True)`
    returns images in [-1, 1] as [B, C, H, W].
    """

    def __init__(
        self,
        num_images: int = 16,
        every_n_steps: int = 10_000,
        nrow: Optional[int] = None,
        log_key: str = "train/image_grid",
        use_ema: bool = True,
    ):
        super().__init__()
        self.num_images = num_images
        self.every_n_steps = every_n_steps
        self.nrow = nrow or int(math.sqrt(num_images))
        self.log_key = log_key
        self.use_ema = use_ema

        self._last_step = 0  # internal counter
        self._progress_bar = None

    @contextmanager
    def _image_progress(self, enabled: bool = True):
        if not enabled:
            yield None
            return

        try:
            from tqdm.auto import tqdm
        except ImportError:
            yield None
            return

        total = max(1, self.num_images)
        self._progress_bar = tqdm(
            total=total,
            desc="Image samples",
            leave=False,
            unit="img",
        )

        try:
            yield self._progress_bar
        finally:
            self._progress_bar.close()
            self._progress_bar = None

    def _generate_images(
        self,
        pl_module: pl.LightningModule,
        progress=None,
    ) -> torch.Tensor:
        """
        Uses the module's sampling API (same assumption as your FID callback).
        """
        device = pl_module.device

        with torch.no_grad():
            imgs = pl_module.generate_samples(  # [-1,1], [B,C,H,W]
                batch_size=self.num_images,
                use_ema=self.use_ema,
            ).to(device)

        if progress is not None:
            progress.update(imgs.shape[0])

        # Map from [-1,1] -> [0,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0
        imgs = imgs.clamp(0.0, 1.0)

        return imgs

    def _log_grid(self, trainer: pl.Trainer, grid: torch.Tensor, global_step: int):
        logger = trainer.logger
        if logger is None:
            return

        # TensorBoard-like logger
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
            logger.experiment.add_image(
                self.log_key,
                grid,              # [C,H,W] tensor in [0,1]
                global_step=global_step,
            )
            return

        # WandB
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
            try:
                import wandb

                logger.experiment.log(
                    {
                        self.log_key: wandb.Image(
                            grid, caption=f"step {global_step}"
                        ),
                        "global_step": global_step,
                    }
                )
                return
            except ImportError:
                pass

        # Generic Lightning log_image API (e.g. some custom loggers)
        try:
            logger.log_image(
                key=self.log_key,
                images=[grid],
                caption=[f"step {global_step}"],
            )
        except Exception:
            # If the logger doesn't support images, just fail silently.
            pass

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:

        # Only main process
        if not trainer.is_global_zero:
            return

        global_step = trainer.global_step

        # Trigger only every N steps
        if global_step - self._last_step < self.every_n_steps:
            return

        self._last_step = global_step

        # Switch to eval for sampling, then restore mode
        was_training = pl_module.training
        pl_module.eval()

        with self._image_progress(enabled=trainer.is_global_zero) as progress:
            imgs = self._generate_images(pl_module, progress)        # [B,C,H,W] in [0,1]

        grid = make_grid(imgs, nrow=self.nrow)         # [C,H,W]
        grid = grid.detach().cpu()

        # Restore train mode
        if was_training:
            pl_module.train()

        self._log_grid(trainer, grid, global_step)
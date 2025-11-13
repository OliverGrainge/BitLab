from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    from bitlab.bittrainer.diffusion.bitimagediffusiontrainer import (
        BitImageDiffusionTrainer,
    )


class InceptionScoreCallback(pl.Callback):
    """
    Compute Inception Score every N training steps using torchmetrics.
    Assumes pl_module.generate_samples(batch_size, use_ema=True) -> [-1,1] images.
    """

    def __init__(
        self,
        num_gen: int = 50_000,
        batch_size: int = 128,
        every_n_steps: int = 10_000,
        splits: int = 10,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_gen = num_gen
        self.batch_size = batch_size
        self.every_n_steps = every_n_steps
        self.splits = splits
        self.normalize = normalize

        self._last_step = 0
        self._progress_bar = None

    @contextmanager
    def _is_progress(self, enabled: bool = True):
        if not enabled:
            yield None
            return

        try:
            from tqdm.auto import tqdm
        except ImportError:
            yield None
            return

        self._progress_bar = tqdm(
            total=self.num_gen,
            desc="IS sampling",
            leave=False,
            unit="img",
        )

        try:
            yield self._progress_bar
        finally:
            self._progress_bar.close()
            self._progress_bar = None

    def _generate_batch(
        self,
        pl_module: "BitImageDiffusionTrainer",
        batch_size: int,
        device: torch.device,
        progress_bar=None,
    ) -> torch.Tensor:
        with torch.no_grad():
            imgs = pl_module.generate_samples(
                batch_size=batch_size,
                use_ema=True,
            )  # [-1,1], [B,C,H,W]

            # map to [0,1]
            imgs = (imgs.clamp(-1, 1) + 1) / 2
            imgs = imgs.clamp(0, 1)

        if progress_bar is not None:
            progress_bar.update(batch_size)

        return imgs.to(device)

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

        try:
            from torchmetrics.image.inception import InceptionScore
        except ImportError as e:
            # Fail loudly; this is a config error, not a silent skip
            print(
                f"[InceptionScoreCallback] torchmetrics not installed or "
                f"no InceptionScore available: {e}"
            )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fresh metric instance per evaluation
        is_metric = InceptionScore(
            splits=self.splits,
            normalize=self.normalize,  # assumes input is [0,1] if True
        ).to(device)

        remaining = self.num_gen

        # Optional: make sure no grads / eval mode for generation
        was_training = pl_module.training
        pl_module.eval()

        with self._is_progress(enabled=trainer.is_global_zero) as progress:
            while remaining > 0:
                b = min(self.batch_size, remaining)
                imgs = self._generate_batch(pl_module, b, device, progress_bar=progress)
                is_metric.update(imgs)
                remaining -= b

        # restore training mode
        if was_training:
            pl_module.train()

        score_mean, score_std = is_metric.compute()  # tensors

        pl_module.log(
            "train/IS_mean",
            score_mean,
            prog_bar=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
        pl_module.log(
            "train/IS_std",
            score_std,
            prog_bar=False,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
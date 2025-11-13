from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    from bitlab.bittrainer.diffusion.bitimagediffusiontrainer import (
        BitImageDiffusionTrainer,
    )


class FIDAndInceptionCallback(pl.Callback):
    """
    Compute Clean-FID and Inception Score every N training steps
    using the *same* generated samples.

    Assumes pl_module.generate_samples(batch_size, use_ema=True) -> [-1,1] images.
    """

    def __init__(
        self,
        # FID config
        dataset_name: str,
        dataset_res: int,
        dataset_split: str = "train",
        mode: str = "clean",
        num_gen: int = 50_000,
        batch_size: int = 128,
        every_n_steps: int = 10_000,
        # IS config
        is_splits: int = 10,
        is_normalize: bool = True,
        compute_is: bool = True,
    ):
        super().__init__()
        # FID params
        self.dataset_name = dataset_name
        self.dataset_res = dataset_res
        self.dataset_split = dataset_split
        self.mode = mode
        self.num_gen = num_gen
        self.batch_size = batch_size
        self.every_n_steps = every_n_steps

        # IS params
        self.is_splits = is_splits
        self.is_normalize = is_normalize
        self.compute_is = compute_is

        self._last_step = 0
        self._progress_bar = None

    @contextmanager
    def _progress(self, enabled: bool = True):
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
            desc="FID+IS sampling",
            leave=False,
            unit="img",
        )

        try:
            yield self._progress_bar
        finally:
            self._progress_bar.close()
            self._progress_bar = None

    def _make_gen_fn(
        self,
        pl_module: "BitImageDiffusionTrainer",
        is_metric,               # may be None if compute_is=False
        is_device: torch.device, # device for IS metric
        progress_bar=None,
    ):
        """
        Generator function used by clean-fid.

        Side effect: updates InceptionScore for each batch if is_metric is not None.
        """

        def gen(z):
            if isinstance(z, torch.Tensor):
                b = z.shape[0]
            else:
                b = z.shape[0]

            with torch.no_grad():
                imgs = pl_module.generate_samples(
                    batch_size=b,
                    use_ema=True,
                )  # [-1,1], [B,C,H,W]

                # map to [0,1]
                imgs = (imgs.clamp(-1, 1) + 1) / 2
                imgs = imgs.clamp(0, 1)

            # Update IS on same samples (if requested)
            if is_metric is not None:
                # imgs may already be on GPU; just ensure device match
                is_metric.update(imgs.to(is_device))

            if progress_bar is not None:
                progress_bar.update(b)

            # clean-fid will consume these for FID
            return imgs

        return gen

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

        # Import deps only when needed
        from cleanfid import fid

        if self.compute_is:
            try:
                from torchmetrics.image.inception import InceptionScore
            except ImportError as e:
                print(
                    f"[FIDAndInceptionCallback] torchmetrics not installed or "
                    f"no InceptionScore available: {e}. "
                    f"Will compute only FID."
                )
                InceptionScore = None
        else:
            InceptionScore = None

        # Device for IS metric
        is_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create metric instance if we want IS and have torchmetrics
        is_metric = None
        if self.compute_is and InceptionScore is not None:
            is_metric = InceptionScore(
                splits=self.is_splits,
                normalize=self.is_normalize,  # expects [0,1] input
            ).to(is_device)

        # Optional: eval mode during sampling
        was_training = pl_module.training
        pl_module.eval()

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        with self._progress(enabled=trainer.is_global_zero) as progress:
            gen_fn = self._make_gen_fn(
                pl_module=pl_module,
                is_metric=is_metric,
                is_device=is_device,
                progress_bar=progress,
            )

            fid_score = fid.compute_fid(
                gen=gen_fn,
                dataset_name=self.dataset_name,
                dataset_res=self.dataset_res,
                dataset_split=self.dataset_split,
                mode=self.mode,
                num_gen=self.num_gen,
                batch_size=self.batch_size,
                device=device_str,
                verbose=False,
            )

        # Restore mode
        if was_training:
            pl_module.train()

        # Log FID
        pl_module.log(
            "train/FID_clean",
            fid_score,
            prog_bar=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )

        # Log IS if computed
        if is_metric is not None:
            is_mean, is_std = is_metric.compute()
            pl_module.log(
                "train/IS_mean",
                is_mean,
                prog_bar=True,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )
            pl_module.log(
                "train/IS_std",
                is_std,
                prog_bar=False,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch

if TYPE_CHECKING:
    from bitlab.bittrainer.diffusion.bitimagediffusiontrainer import (
        BitImageDiffusionTrainer,
    )


class CleanFIDCallback(pl.Callback):
    """
    Compute Clean-FID every N training steps using clean-fid's generator API.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_res: int,
        dataset_split: str = "train",
        mode: str = "clean",
        num_gen: int = 50_000,
        batch_size: int = 128,
        every_n_steps: int = 10_000,   # <-- step interval (new)
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_res = dataset_res
        self.dataset_split = dataset_split
        self.mode = mode
        self.num_gen = num_gen
        self.batch_size = batch_size
        self.every_n_steps = every_n_steps

        self._last_step = 0   # internal counter

    def _make_gen_fn(self, pl_module: "BitImageDiffusionTrainer"):
        import numpy as np

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

                imgs = (imgs.clamp(-1, 1) + 1) / 2        # [0,1]
                imgs = (imgs * 255).round().to(torch.uint8)
                imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # [B,H,W,C]

            return imgs.astype(np.uint8)

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

        from cleanfid import fid

        gen_fn = self._make_gen_fn(pl_module)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        score = fid.compute_fid(
            gen=gen_fn,
            dataset_name=self.dataset_name,
            dataset_res=self.dataset_res,
            dataset_split=self.dataset_split,
            mode=self.mode,
            num_gen=self.num_gen,
            batch_size=self.batch_size,
            device=device,
        )

        pl_module.log(
            "train/FID_clean",
            score,
            prog_bar=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
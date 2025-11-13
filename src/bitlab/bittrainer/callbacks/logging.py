from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from pytorch_lightning import Callback
import math
from typing import Optional

import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid



def _iter_named_parameters_by_module_type(
    model: torch.nn.Module,
) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Group named parameters by their module (layer) type."""
    grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}

    for name, module in model.named_modules():
        layer_type = type(module).__name__

        if not hasattr(module, "weight") or module.weight is None:
            continue

        parameter = module.weight
        grouped.setdefault(layer_type, []).append((name if name else "root", parameter))

    return grouped



def _log_histogram(logger, tag: str, values: torch.Tensor, global_step: int) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    values_cpu = values.detach().cpu()

    if hasattr(experiment, "add_histogram"):
        experiment.add_histogram(tag, values_cpu, global_step=global_step)
        return

    try:
        import wandb  # type: ignore
        import numpy as np

        flattened = values_cpu.numpy().ravel()
        if flattened.size == 0:
            return

        unique_bins = max(1, np.unique(flattened).size)
        num_bins = min(64, unique_bins)

        experiment.log(
            {
                "global_step": global_step,
                tag: wandb.Histogram(flattened, num_bins=num_bins),
            }
        )
    except ImportError:
        print(f"Warning: Could not import wandb to log histogram '{tag}'.")
    except ValueError:
        # Skip logging if WandB histogram still fails due to degenerate data.
        pass


def _log_image(
    logger,
    tag: str,
    image: torch.Tensor,
    global_step: int,
    caption: Optional[str] = None,
) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    if hasattr(experiment, "add_image"):
        experiment.add_image(tag, image, global_step=global_step)
        if caption and hasattr(experiment, "add_text"):
            experiment.add_text(tag + "_caption", caption, global_step=global_step)
    else:
        try:
            import wandb  # type: ignore

            kwargs = {"caption": caption} if caption else {}
            experiment.log(
                {
                    "global_step": global_step,
                    tag: wandb.Image(image, **kwargs),
                }
            )
        except ImportError:
            print(f"Warning: Could not import wandb to log image '{tag}'.")


def _split_logits_for_bce(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() > 1 and logits.shape[1] == 2:
        return logits[:, 1]
    if logits.dim() > 1 and logits.shape[1] == 1:
        return logits.squeeze(1)
    return logits


class GradientNormLogger(Callback):
    """Log gradient norms grouped by layer type every N steps."""

    def __init__(self, every_n_steps: int = 100) -> None:
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(
        self, trainer, pl_module, optimizer
    ) -> None:  # noqa: D401
        if trainer.logger is None:
            return
        if self.every_n_steps <= 0:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return

        layer_groups = _iter_named_parameters_by_module_type(pl_module.model)
        for layer_type, param_list in layer_groups.items():
            for layer_name, parameter in param_list:
                if parameter.grad is None:
                    continue

                grad_norm = parameter.grad.norm().item()
                tag = f"gradients/{layer_type}/{layer_name}_norm"
                pl_module.log(tag, grad_norm, on_step=False, on_epoch=True)

        total_norm_sq = 0.0
        for parameter in pl_module.model.parameters():
            if parameter.grad is None:
                continue
            total_norm_sq += parameter.grad.norm().item() ** 2

        total_norm = total_norm_sq**0.5
        pl_module.log(
            "gradients/global_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )


class WeightHistogramLogger(Callback):
    """Log weight histograms grouped by layer type every N training steps."""

    def __init__(self, log_every_n_steps: int = 10_000) -> None:
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive.")
        self.log_every_n_steps = log_every_n_steps
        self._last_step = 0  # internal counter

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Only main process
        if not trainer.is_global_zero:
            return

        if trainer.logger is None:
            return

        global_step = trainer.global_step

        # Trigger only every N steps
        if global_step - self._last_step < self.log_every_n_steps:
            return

        self._last_step = global_step

        layer_groups = _iter_named_parameters_by_module_type(pl_module.model)
        if not layer_groups:
            return

        for layer_type, param_list in layer_groups.items():
            for layer_name, parameter in param_list:
                tag = f"weights/{layer_type}/{layer_type}-{layer_name}"
                _log_histogram(
                    trainer.logger,
                    tag,
                    parameter.data,
                    global_step,  # or pl_module.global_step, but this is equivalent here
                )


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

    def _generate_images(self, pl_module: pl.LightningModule) -> torch.Tensor:
        """
        Uses the module's sampling API (same assumption as your FID callback).
        """
        device = pl_module.device

        with torch.no_grad():
            imgs = pl_module.generate_samples(  # [-1,1], [B,C,H,W]
                batch_size=self.num_images,
                use_ema=self.use_ema,
            ).to(device)

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

        imgs = self._generate_images(pl_module)        # [B,C,H,W] in [0,1]
        grid = make_grid(imgs, nrow=self.nrow)         # [C,H,W]
        grid = grid.detach().cpu()

        # Restore train mode
        if was_training:
            pl_module.train()

        self._log_grid(trainer, grid, global_step)


class ClassificationVisualizationLogger(Callback):
    """Log confusion matrices and sample predictions for classification tasks."""

    def __init__(
        self,
        num_samples_to_log: int = 16,
        log_samples_every_n_epochs: int = 5,
        log_confusion_matrix: bool = True,
    ) -> None:
        if log_samples_every_n_epochs <= 0:
            raise ValueError("log_samples_every_n_epochs must be positive.")

        self.num_samples_to_log = num_samples_to_log
        self.log_samples_every_n_epochs = log_samples_every_n_epochs
        self.log_confusion_matrix = log_confusion_matrix

        self._buffer_images: List[torch.Tensor] = []
        self._buffer_labels: List[torch.Tensor] = []
        self._buffer_preds: List[torch.Tensor] = []

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._buffer_images = []
        self._buffer_labels = []
        self._buffer_preds = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        if len(self._buffer_images) >= self.num_samples_to_log:
            return
        if batch_idx != 0:
            return

        images, targets = batch
        remaining = self.num_samples_to_log - len(self._buffer_images)
        images = images[:remaining]
        targets = targets[:remaining]

        device = pl_module.device
        logits = pl_module(images.to(device))

        if getattr(pl_module, "loss_type", "cross_entropy") == "bce":
            logits = _split_logits_for_bce(logits)
            preds = torch.sigmoid(logits).detach().cpu()
            preds = (preds > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1).detach().cpu()

        self._buffer_images.append(images.detach().cpu())
        self._buffer_labels.append(targets.detach().cpu())
        self._buffer_preds.append(preds)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.logger is None:
            return

        if self.log_confusion_matrix and hasattr(pl_module, "val_confusion_matrix"):
            try:
                cm = pl_module.val_confusion_matrix.compute()
            except ValueError:
                cm = None

            if cm is not None:
                self._log_confusion_matrix(trainer.logger, cm, pl_module.global_step)

        if trainer.current_epoch % self.log_samples_every_n_epochs != 0:
            return

        if not self._buffer_images:
            return

        images = torch.cat(self._buffer_images, dim=0)[: self.num_samples_to_log]
        labels = torch.cat(self._buffer_labels, dim=0)[: self.num_samples_to_log]
        preds = torch.cat(self._buffer_preds, dim=0)[: self.num_samples_to_log]

        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)

        grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)

        caption_lines = [
            f"Sample {i}: True={labels[i].item()}, Pred={preds[i].item()}"
            for i in range(min(8, len(labels)))
        ]
        caption = "\n".join(caption_lines)

        _log_image(
            trainer.logger, "val/samples", grid, pl_module.global_step, caption=caption
        )

    @staticmethod
    def _log_confusion_matrix(
        logger, confusion_matrix: torch.Tensor, global_step: int
    ) -> None:
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            return

        cm_cpu = confusion_matrix.detach().cpu()

        if hasattr(experiment, "add_figure"):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(cm_cpu.numpy(), cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")

            experiment.add_figure("val/confusion_matrix", fig, global_step=global_step)
            plt.close(fig)
        else:
            try:
                import wandb  # type: ignore

                experiment.log(
                    {
                        "global_step": global_step,
                        "val/confusion_matrix": wandb.Image(cm_cpu.numpy()),
                    }
                )
            except ImportError:
                print("Warning: Could not import wandb to log confusion matrix.")

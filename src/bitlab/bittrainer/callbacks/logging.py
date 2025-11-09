from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from pytorch_lightning import Callback


def _iter_named_parameters_by_module_type(model: torch.nn.Module) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Group named parameters by their module (layer) type."""
    grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}

    for name, module in model.named_modules():
        layer_type = type(module).__name__

        if not hasattr(module, "weight") or module.weight is None:
            continue

        parameter = module.weight
        grouped.setdefault(layer_type, []).append((name if name else "root", parameter))

    return grouped


def _log_to_experiment(logger, tag: str, value, global_step: int) -> None:
    """Log scalar values directly via the underlying experiment if available."""
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    if hasattr(experiment, "add_scalar"):
        experiment.add_scalar(tag, value, global_step=global_step)
    else:
        try:
            import wandb  # type: ignore

            if isinstance(value, torch.Tensor):
                value = value.item()
            experiment.log({"global_step": global_step, tag: value})
        except ImportError:
            print(f"Warning: Could not import wandb to log scalar '{tag}'.")


def _log_histogram(logger, tag: str, values: torch.Tensor, global_step: int) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    values_cpu = values.detach().cpu()

    if hasattr(experiment, "add_histogram"):
        experiment.add_histogram(tag, values_cpu, global_step=global_step)
    else:
        try:
            import wandb  # type: ignore

            experiment.log(
                {
                    "global_step": global_step,
                    tag: wandb.Histogram(values_cpu.numpy()),
                }
            )
        except ImportError:
            print(f"Warning: Could not import wandb to log histogram '{tag}'.")


def _log_image(logger, tag: str, image: torch.Tensor, global_step: int, caption: Optional[str] = None) -> None:
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

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:  # noqa: D401
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

        total_norm = total_norm_sq ** 0.5
        pl_module.log("gradients/global_norm", total_norm, on_step=True, on_epoch=True, prog_bar=False)


class WeightHistogramLogger(Callback):
    """Log weight histograms grouped by layer type every N validation epochs."""

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        if log_every_n_epochs <= 0:
            raise ValueError("log_every_n_epochs must be positive.")
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.logger is None:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        layer_groups = _iter_named_parameters_by_module_type(pl_module.model)
        if not layer_groups:
            return

        for layer_type, param_list in layer_groups.items():
            for layer_name, parameter in param_list:
                tag = f"weights/{layer_type}/{layer_type}-{layer_name}"
                _log_histogram(trainer.logger, tag, parameter.data, pl_module.global_step)


class DiffusionSampleLogger(Callback):
    """Generate DDIM samples and log statistics/images during validation."""

    def __init__(
        self,
        batch_size: int,
        num_steps: int,
        log_every_n_epochs: int = 1,
        use_ema: Optional[bool] = None,
    ) -> None:
        if log_every_n_epochs <= 0:
            raise ValueError("log_every_n_epochs must be positive.")

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.use_ema = use_ema

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.logger is None:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if not hasattr(pl_module, "sample_ddim"):
            return

        sample_kwargs = {
            "batch_size": self.batch_size,
            "num_steps": self.num_steps,
        }
        if self.use_ema is not None:
            sample_kwargs["use_ema"] = self.use_ema

        samples = pl_module.sample_ddim(**sample_kwargs)

        sample_means = samples.mean(dim=[1, 2, 3])
        sample_stds = samples.std(dim=[1, 2, 3])
        avg_sample_mean = sample_means.mean().item()
        avg_sample_std = sample_stds.mean().item()
        inter_sample_variance = samples.var(dim=0).mean().item()
        global_mean = samples.mean().item()
        global_std = samples.std().item()

        stats = {
            "samples/mean": avg_sample_mean,
            "samples/std": avg_sample_std,
            "samples/inter_sample_variance": inter_sample_variance,
            "samples/global_mean": global_mean,
            "samples/global_std": global_std,
        }

        for tag, value in stats.items():
            pl_module.log(tag, value, sync_dist=True)

        normalized = (samples + 1.0) / 2.0
        normalized = torch.clamp(normalized, 0.0, 1.0)
        grid = torchvision.utils.make_grid(normalized, nrow=4, normalize=False)

        _log_image(trainer.logger, "samples", grid, pl_module.global_step)


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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
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

        _log_image(trainer.logger, "val/samples", grid, pl_module.global_step, caption=caption)

    @staticmethod
    def _log_confusion_matrix(logger, confusion_matrix: torch.Tensor, global_step: int) -> None:
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


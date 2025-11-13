import torch 
from typing import Dict, List, Tuple
from pytorch_lightning.callbacks import Callback

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
from typing import Dict, List, Tuple
import torch
from pytorch_lightning import Callback

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
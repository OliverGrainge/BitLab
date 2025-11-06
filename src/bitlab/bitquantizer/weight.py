"""Weight quantization functions."""
from typing import Tuple
import torch


def quantize_weight_wpt(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using wpt scheme (shared between ai8pc and ai8pg)."""
    qws = w.abs().mean()
    qw = (w / (qws + eps)).round().clamp(-1, 1)
    return qws, qw


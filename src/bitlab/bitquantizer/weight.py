"""Weight quantization functions."""
from typing import Tuple
import torch


def quantize_weight_wpt(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using wpt scheme.
    
    Supports both linear (2D) and convolutional (4D) tensors:
    - 2D [out_features, in_features]: per-tensor quantization
    - 4D [out_ch, in_ch, kh, kw]: per-output-channel quantization
    """
    if w.ndim == 2:  # Linear: [out_features, in_features]
        # Per-tensor quantization (global mean)
        qws = w.abs().mean()
    elif w.ndim == 4:  # Conv: [out_channels, in_channels, kernel_h, kernel_w]
        # Per-output-channel quantization
        qws = w.abs().mean(dim=(1, 2, 3), keepdim=True)
    else:
        raise ValueError(f"Unsupported weight tensor dimension: {w.ndim}. Expected 2D or 4D.")
    
    qw = (w / (qws + eps)).round().clamp(-1, 1)
    return qws, qw


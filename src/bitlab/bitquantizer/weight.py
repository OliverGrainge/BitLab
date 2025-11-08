"""Weight quantization functions."""
from typing import Tuple
import torch


def quantize_weight_wpt(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using wpt (ternary per-tensor, global) scheme.
    
    Single scale for entire tensor regardless of 2D or 4D.
    """
    qws = w.abs().mean()  # Global mean for all dims
    qw = (w / (qws + eps)).round().clamp(-1, 1)
    return qws, qw


def quantize_weight_wpc(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using wpc (ternary per-channel) scheme.
    
    - 2D [out_features, in_features]: per-output-feature (per-row)
    - 4D [out_ch, in_ch, kh, kw]: per-output-channel
    """
    if w.ndim == 2:
        qws = w.abs().mean(dim=1, keepdim=True)
    elif w.ndim == 4:
        qws = w.abs().mean(dim=(1, 2, 3), keepdim=True)
    else:
        raise ValueError(f"Unsupported weight dimension: {w.ndim}")
    
    qw = (w / (qws + eps)).round().clamp(-1, 1)
    return qws, qw


def quantize_weight_wpg(
    w: torch.Tensor, eps: float = 1e-6, group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using wpg (ternary per-group) scheme.
    
    Args:
        w: Weight tensor [out_features, in_features]
        group_size: Size of each group along in_features
    """
    assert w.ndim == 2, "Per-group only supports 2D linear weights"
    
    out_features, in_features = w.shape
    
    # Pad if needed
    pad_size = (group_size - in_features % group_size) % group_size
    if pad_size > 0:
        w_padded = torch.nn.functional.pad(w, (0, pad_size))
    else:
        w_padded = w
    
    # Reshape to groups: [out_features, num_groups, group_size]
    w_grouped = w_padded.view(out_features, -1, group_size)
    
    # Per-group abs-mean
    qws = w_grouped.abs().mean(dim=2, keepdim=True)
    
    # Quantize
    qw_grouped = (w_grouped / (qws + eps)).round().clamp(-1, 1)
    qw = qw_grouped.view(out_features, -1)[:, :in_features]  # Remove padding
    qws = qws.squeeze(-1)  # [out_features, num_groups]
    
    return qws, qw



def quantize_weight_wbf16(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert to bfloat16
    qw = w.to(torch.bfloat16)
    # Return dummy scale of 1.0 for API consistency
    qws = torch.tensor(1.0, dtype=torch.bfloat16, device=w.device)
    return qws, qw


def quantize_weight_wf16(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    qw = w.to(torch.float16)
    # Return dummy scale of 1.0 for API consistency
    qws = torch.tensor(1.0, dtype=torch.float16, device=w.device)
    return qws, qw


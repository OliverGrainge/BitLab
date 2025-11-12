"""Weight quantization functions."""

from typing import Tuple

import torch


def quantize_weight_wpt(
    w: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = w.dtype
    w_fp32 = w.float()
    scale = w_fp32.abs().mean().clamp(min=eps)  # Global mean for all dims
    inv_scale = 1.0 / scale
    qw = (w_fp32 * inv_scale).round().clamp(-1, 1)
    return inv_scale.to(dtype=orig_dtype), qw.to(dtype=orig_dtype)


def quantize_weight_wpc(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using wpc (ternary per-channel) scheme.

    - 2D [out_features, in_features]: per-output-feature (per-row)
    - 4D [out_ch, in_ch, kh, kw]: per-output-channel
    """
    if w.ndim == 2:
        scale = w.abs().mean(dim=1, keepdim=True)
    elif w.ndim == 4:
        scale = w.abs().mean(dim=(1, 2, 3), keepdim=True)
    else:
        raise ValueError(f"Unsupported weight dimension: {w.ndim}")

    scale = scale.clamp(min=eps)
    inv_scale = 1.0 / scale
    qw = (w * inv_scale).round().clamp(-1, 1)
    return inv_scale, qw


def quantize_weight_wpg(
    w: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
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
    scale = w_grouped.abs().mean(dim=2, keepdim=True).clamp(min=eps)

    # Quantize
    inv_scale = 1.0 / scale
    qw_grouped = (w_grouped * inv_scale).round().clamp(-1, 1)
    qw = qw_grouped.view(out_features, -1)[:, :in_features]  # Remove padding
    inv_scale = inv_scale.squeeze(-1)  # [out_features, num_groups]

    return inv_scale, qw


def quantize_weight_wbf16(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert to bfloat16
    qw = w.to(torch.bfloat16)
    # Return dummy inverse scale of 1.0 for API consistency
    inv_scale = torch.tensor(1.0, dtype=torch.bfloat16, device=w.device)
    return inv_scale, qw


def quantize_weight_wf16(
    w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    qw = w.to(torch.float16)
    # Return dummy inverse scale of 1.0 for API consistency
    inv_scale = torch.tensor(1.0, dtype=torch.float16, device=w.device)
    return inv_scale, qw

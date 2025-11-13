import math
from typing import Any, ClassVar, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitmodels.auto import register_bitmodel
from bitlab.bitmodels.base import BaseBitModel
from bitlab.bitmodels.imagegeneration.unet.config import BitUNetConfig
from bitlab.bitmodels.mixins import ImageGenerationMixin


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Applied after each BitConv2d layer for stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS norm
        # For Conv2d outputs with shape [B, C, H, W], we normalize over C dimension
        if x.dim() == 4:  # Conv2d output
            # Compute RMS over spatial dimensions for each channel
            norm = x.pow(2).mean(dim=[2, 3], keepdim=True).sqrt() + self.eps
            x = x / norm
            # Apply learnable weight per channel
            return x * self.weight.view(1, -1, 1, 1)
        else:  # Linear output [B, D]
            norm = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
            x = x / norm
            return x * self.weight


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class BitConvBlock(nn.Module):
    """Wrapper for RMSNorm + BitConv2d combination.

    Following the BitLinear pattern: RMSNorm is applied to activations
    BEFORE the BitConv2d layer.
    Always uses quantized BitConv2d; quant_type selectable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        quant_type: str = "ai8pc_wpt",
        apply_norm: bool = True,  # Option to skip norm for certain layers
    ):
        super().__init__()
        self.apply_norm = apply_norm
        # Import BitConv2d lazily
        from bitcore.bnn.bitlayers import BitConv2d

        if apply_norm:
            self.norm = RMSNorm(in_channels)
        else:
            self.norm = nn.Identity()

        self.conv = BitConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            quant_type=quant_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply normalization to activations first (RMSNorm)
        x = self.norm(x)
        # Then apply BitConv2d
        x = self.conv(x)
        return x


class BitResidualBlock(nn.Module):
    """Residual block with time embedding and quantized BitConv2d usage.

    RMSNorm is applied before BitConv2d layers where appropriate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        quant_type: str = "ai8pc_wpt",
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        # First GroupNorm (separate from quantization norms)
        self.norm1 = nn.GroupNorm(32, in_channels)

        # First quantized conv
        self.conv1 = BitConvBlock(
            in_channels,
            out_channels,
            3,
            padding=1,
            quant_type=quant_type,
            apply_norm=False,
        )

        # Second GroupNorm
        self.norm2 = nn.GroupNorm(32, out_channels)

        # Second quantized conv
        self.conv2 = BitConvBlock(
            out_channels,
            out_channels,
            3,
            padding=1,
            quant_type=quant_type,
            apply_norm=False,
        )

        # Explicit RMSNorm layers for activations (applied before BitConv)
        self.act_norm1 = RMSNorm(in_channels)
        self.act_norm2 = RMSNorm(out_channels)

        # Time embedding projection - keep as regular linear
        self.time_emb_proj = nn.Linear(
            time_emb_dim, out_channels * 2 if use_scale_shift_norm else out_channels
        )

        self.dropout = nn.Dropout(dropout)

        # Skip connection (quantized 1x1 conv if channels differ)
        if in_channels != out_channels:
            self.skip_connection = BitConvBlock(
                in_channels, out_channels, 1, quant_type=quant_type, apply_norm=True
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        # Apply RMSNorm to activations before quantized conv
        h = self.act_norm1(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = F.silu(time_emb)
        time_emb = self.time_emb_proj(time_emb)[:, :, None, None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = h + time_emb
            h = self.norm2(h)

        h = F.silu(h)
        h = self.dropout(h)
        # Apply RMSNorm to activations before second quantized conv
        h = self.act_norm2(h)
        h = self.conv2(h)

        return h + self.skip_connection(x)


class BitAttentionBlock(nn.Module):
    """Self-attention block using quantized projections (BitConv2d)."""

    def __init__(
        self, channels: int, num_heads: int = 4, quant_type: str = "ai8pc_wpt"
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)

        # Use quantized convolutions for QKV and output projections
        self.qkv = BitConvBlock(channels, channels * 3, 1, quant_type=quant_type)
        self.proj_out = BitConvBlock(channels, channels, 1, quant_type=quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj_out(out)
        return out + residual


class BitDownsampleBlock(nn.Module):
    """Downsampling block implemented with quantized BitConv2d (stride=2)."""

    def __init__(self, channels: int, quant_type: str = "ai8pc_wpt"):
        super().__init__()
        self.conv = BitConvBlock(
            channels, channels, 3, stride=2, padding=1, quant_type=quant_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BitUpsampleBlock(nn.Module):
    """Upsampling block implemented with quantized BitConv2d."""

    def __init__(self, channels: int, quant_type: str = "ai8pc_wpt"):
        super().__init__()
        self.conv = BitConvBlock(
            channels, channels, 3, padding=1, quant_type=quant_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class BitUNet(nn.Module):
    """Quantized (Bit) U-Net architecture for the denoising network."""

    def __init__(
        self,
        config, 
    ):
        super().__init__()

        self.in_channels = config.in_channels
        self.model_channels = config.model_channels
        self.num_res_blocks = config.num_res_blocks
        self.attention_resolutions = config.attention_resolutions
        self.channel_mult = config.channel_mult
        self.out_channels = config.out_channels
        self.quant_type = config.quant_type
        self.dropout = config.dropout 
        self.num_heads = config.num_heads
        self.time_emb_dim = config.model_channels * 4
        self.use_scale_shift_norm = config.use_scale_shift_norm

        # Time embedding (keep as regular layers)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(self.model_channels),
            nn.Linear(self.model_channels, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        # Initial quantized convolution
        self.conv_in = BitConvBlock(
            self.in_channels, self.model_channels, 3, padding=1, quant_type=self.quant_type
        )

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = self.model_channels
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    BitResidualBlock(
                        ch,
                        mult * self.model_channels,
                        self.time_emb_dim,
                        self.dropout,
                        self.use_scale_shift_norm,
                        quant_type=self.quant_type,
                    )
                ]
                ch = mult * self.model_channels

                if ds in self.attention_resolutions:
                    layers.append(
                        BitAttentionBlock(ch, self.num_heads, quant_type=self.quant_type)
                    )

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(self.channel_mult) - 1:
                self.down_blocks.append(
                    nn.ModuleList([BitDownsampleBlock(ch, quant_type=self.quant_type)])
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle blocks
        self.middle_blocks = nn.ModuleList(
            [
                BitResidualBlock(
                    ch,
                    ch,
                    self.time_emb_dim,
                    self.dropout,
                    self.use_scale_shift_norm,
                    quant_type=self.quant_type,
                ),
                BitAttentionBlock(ch, self.num_heads, quant_type=self.quant_type),
                BitResidualBlock(
                    ch,
                    ch,
                    self.time_emb_dim,
                    self.dropout,
                    self.use_scale_shift_norm,
                    quant_type=self.quant_type,
                ),
            ]
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(self.channel_mult)):
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    BitResidualBlock(
                        ch + ich,
                        self.model_channels * mult,
                        self.time_emb_dim,
                        self.dropout,
                        self.use_scale_shift_norm,
                        quant_type=self.quant_type,
                    )
                ]
                ch = self.model_channels * mult

                if ds in self.attention_resolutions:
                    layers.append(
                        BitAttentionBlock(ch, self.num_heads, quant_type=self.quant_type)
                    )

                if level != len(self.channel_mult) - 1 and i == self.num_res_blocks:
                    layers.append(BitUpsampleBlock(ch, quant_type=self.quant_type))
                    ds //= 2

                self.up_blocks.append(nn.ModuleList(layers))

        # Output (quantized)
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            BitConvBlock(ch, self.out_channels, 3, padding=1, quant_type=self.quant_type),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Initial convolution
        h = self.conv_in(x)

        # Downsampling
        hs = [h]
        for modules in self.down_blocks:
            for module in modules:
                if isinstance(module, BitResidualBlock):
                    h = module(h, time_emb)
                else:
                    h = module(h)
            hs.append(h)

        # Middle
        for module in self.middle_blocks:
            if isinstance(module, BitResidualBlock):
                h = module(h, time_emb)
            else:
                h = module(h)

        # Upsampling
        for modules in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for module in modules:
                if isinstance(module, BitResidualBlock):
                    h = module(h, time_emb)
                else:
                    h = module(h)

        return self.out(h)


@register_bitmodel("bitunet")
class BitUNetModel(ImageGenerationMixin, BaseBitModel):
    """
    Bit U-Net Model (quantized)

    Always uses BitConv2d-based blocks (quantization enabled). The `quant_type`
    remains configurable so callers can select the quantization type.
    """

    config_cls: ClassVar[type[BitUNetConfig]] = BitUNetConfig

    def __init__(
        self,
        config: Optional[BitUNetConfig] = None,
        quant_type: Optional[str] = None,
        **overrides: Any,
    ):
        updates: dict[str, Any] = dict(overrides)
        if quant_type is not None:
            updates.setdefault("quant_type", quant_type)

        super().__init__(config=config, **updates)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps)

    def build_model(self, config: BitUNetConfig) -> nn.Module:
        return BitUNet(config)

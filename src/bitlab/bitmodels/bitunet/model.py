import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitmodels.auto import register_bitmodel
from bitlab.bitmodels.unet.config import BitUNetConfig


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


class ResidualBlock(nn.Module):
    """Residual block with time embedding and optional conditional input."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels * 2 if use_scale_shift_norm else out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
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
        h = self.conv2(h)
        
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
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
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        out = self.proj_out(out)
        return out + residual


class DownsampleBlock(nn.Module):
    """Downsampling block using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Upsampling block using nearest neighbor interpolation + convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for the denoising network."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float,
        channel_mult: Tuple[int, ...],
        num_heads: int,
        time_emb_dim: int,
        use_scale_shift_norm: bool
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch, 
                        mult * model_channels, 
                        time_emb_dim, 
                        dropout, 
                        use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([DownsampleBlock(ch)]))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm)
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResidualBlock(
                        ch + ich,
                        model_channels * mult,
                        time_emb_dim,
                        dropout,
                        use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                if level != len(channel_mult) - 1 and i == num_res_blocks:
                    layers.append(UpsampleBlock(ch))
                    ds //= 2
                
                self.up_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
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
                if isinstance(module, ResidualBlock):
                    h = module(h, time_emb)
                else:
                    h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.middle_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            else:
                h = module(h)
        
        # Upsampling
        for modules in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for module in modules:
                if isinstance(module, ResidualBlock):
                    h = module(h, time_emb)
                else:
                    h = module(h)
        
        return self.out(h)


@register_bitmodel("bitunet")
class BitUNetModel(nn.Module):
    """
    Pixel Space Diffusion U-Net Model
    
    A U-Net architecture for diffusion models that operates in pixel space.
    This is just the model architecture - diffusion logic should be handled by the trainer.
    
    Args:
        image_size: Size of input images (assumes square images)
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (typically same as in_channels)
        model_channels: Base channel count for the U-Net
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply attention (e.g., (1, 2, 4))
        dropout: Dropout probability
        channel_mult: Channel multiplier for each resolution level (e.g., (1, 2, 4, 8))
        num_heads: Number of attention heads
        use_scale_shift_norm: Whether to use scale+shift normalization in residual blocks
    """
    
    def __init__(
        self,
        config: Optional[BitUNetConfig] = None,
        **overrides,
    ):
        super().__init__()

        if config is None:
            config = BitUNetConfig(**overrides)
        else:
            if not isinstance(config, BitUNetConfig):
                raise TypeError(
                    "config must be a BitUNetConfig instance or None"
                )
            if overrides:
                config = config.with_overrides(**overrides)

        self.config = config

        # Cache frequently used fields for convenience
        in_channels = config.in_channels
        out_channels = config.out_channels
        model_channels = config.model_channels
        num_res_blocks = config.num_res_blocks
        attention_resolutions = config.attention_resolutions
        dropout = config.dropout
        channel_mult = config.channel_mult
        num_heads = config.num_heads
        use_scale_shift_norm = config.use_scale_shift_norm

        # Time embedding dimension
        time_emb_dim = model_channels * 4

        # Create the denoising U-Net
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads,
            time_emb_dim=time_emb_dim,
            use_scale_shift_norm=use_scale_shift_norm,
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input images (noisy images) [B, C, H, W]
            timesteps: Diffusion timesteps [B]
        
        Returns:
            Model output [B, C, H, W]
        """
        return self.model(x, timesteps)
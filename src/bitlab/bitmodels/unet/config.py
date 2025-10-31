from typing import Literal, Tuple

from pydantic import Field

from bitlab.bitmodels.config import BaseBitModelConfig, register_bitconfig


@register_bitconfig("bitunet")
class BitUNetConfig(BaseBitModelConfig):
    """
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
    model_type: Literal["bitunet"] = Field(default="bitunet", frozen=True)
    image_size: int = Field(default=64)
    in_channels: int = Field(default=3)
    out_channels: int = Field(default=3)
    model_channels: int = Field(default=128)
    num_res_blocks: int = Field(default=2)
    attention_resolutions: Tuple[int, ...] = Field(default=(1, 2, 4))
    dropout: float = Field(default=0.0)
    channel_mult: Tuple[int, ...] = Field(default=(1, 2, 3, 4))
    num_heads: int = Field(default=4)
    use_scale_shift_norm: bool = Field(default=True)

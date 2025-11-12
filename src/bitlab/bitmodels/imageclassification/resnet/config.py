from typing import Literal, Tuple

from pydantic import Field

from bitlab.bitmodels.config import BaseBitModelConfig, register_bitconfig
from bitlab.bitmodels.tasks import ModelTask


@register_bitconfig("bitresnet")
class BitResNetConfig(BaseBitModelConfig):
    """
    num_classes: Number of output classes.
    in_channels: Number of input channels (e.g., 3 for RGB).
    base_channels: Number of channels in the first convolution.
    block_layers: Number of residual blocks in each ResNet stage.
    use_bitconv: Whether to instantiate convolutions with `BitConv2d`.
    quant_type: Quantization type identifier when using `BitConv2d`.
    """

    task: Literal[ModelTask.IMAGE_CLASSIFICATION.value] = Field(
        default=ModelTask.IMAGE_CLASSIFICATION.value, frozen=True
    )
    model_type: Literal["bitresnet"] = Field(default="bitresnet", frozen=True)
    num_classes: int = Field(default=10)
    in_channels: int = Field(default=3)
    base_channels: int = Field(default=64)
    block_layers: Tuple[int, int, int, int] = Field(default=(2, 2, 2, 2))
    quant_type: str | None = Field(default=None)

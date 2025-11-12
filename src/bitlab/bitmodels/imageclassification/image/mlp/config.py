from typing import Literal, Tuple

from pydantic import Field

from bitlab.bitmodels.config import BaseBitModelConfig, register_bitconfig
from bitlab.bitmodels.tasks import ModelTask


@register_bitconfig("bitmlp")
class BitMLPConfig(BaseBitModelConfig):
    """
    input_size: Flattened input dimensionality (defaults to 28x28 images).
    hidden_dims: Tuple specifying hidden layer sizes for the MLP.
    num_classes: Number of output classes.
    use_bitlinear: Whether to construct hidden layers with `BitLinear`.
    quant_type: Quantization type identifier when using `BitLinear`.
    """

    task: Literal[ModelTask.IMAGE_CLASSIFICATION.value] = Field(
        default=ModelTask.IMAGE_CLASSIFICATION.value, frozen=True
    )
    model_type: Literal["bitmlp"] = Field(default="bitmlp", frozen=True)
    input_size: int = Field(default=28 * 28)
    hidden_dims: Tuple[int, ...] = Field(default=(256, 256))
    num_classes: int = Field(default=10)
    quant_type: str | None = Field(default=None)

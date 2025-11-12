from __future__ import annotations

from functools import partial
from typing import Any, ClassVar, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitmodels.auto import register_bitmodel
from bitlab.bitmodels.base import BaseBitModel
from bitlab.bitmodels.imageclassification.resnet.config import \
    BitResNetConfig
from bitlab.bitmodels.mixins import ImageClassificationMixin
from bitlab.bnn.bitlayers import BitConv2d


class BasicBlock(nn.Module):
    """ResNet basic block that can swap convolution implementations."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        conv_factory,
    ) -> None:
        super().__init__()
        self.conv1 = conv_factory(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_factory(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_factory(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class BitResNet(nn.Module):
    """ResNet-18 style architecture with optional BitConv2d layers."""

    def __init__(
        self,
        config, 
    ) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.base_channels = config.base_channels
        self.block_layers = config.block_layers
        self.quant_type = config.quant_type

        use_bitconv = self.quant_type is not None

        if use_bitconv:
            conv_factory = partial(BitConv2d, quant_type=self.quant_type)
        else:
            conv_factory = nn.Conv2d

        in_channels = self.in_channels
        base_channels = self.base_channels
        num_classes = self.num_classes
        block_layers = self.block_layers

        def make_conv(
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
        ):
            return conv_factory(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

        self.conv_factory = make_conv

        self.in_planes = base_channels
        self.conv1 = self.conv_factory(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        layers = list(block_layers)
        if len(layers) != 4:
            raise ValueError("block_layers must contain 4 integers for ResNet stages")

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

        self.fc = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.conv_factory))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


@register_bitmodel("bitresnet")
class BitResNetModel(ImageClassificationMixin, BaseBitModel):
    """BitResNetModel wraps the ResNet architecture with config support."""

    config_cls: ClassVar[type[BitResNetConfig]] = BitResNetConfig

    def __init__(
        self, config: Optional[BitResNetConfig] = None, **overrides: Any
    ) -> None:
        super().__init__(config=config, **overrides)

    def build_model(self, config: BitResNetConfig) -> nn.Module:
        return BitResNet(config=config)

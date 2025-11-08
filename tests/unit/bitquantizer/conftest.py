"""Shared fixtures and helpers for bitquantizer unit tests."""
from __future__ import annotations

from math import ceil
from typing import Tuple

import pytest
import torch

from bitlab.bitquantizer.registry import (
    ACT_QUANT_REGISTRY,
    QUANTIZER_REGISTRY,
    WEIGHT_QUANT_REGISTRY,
)

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------




@pytest.fixture
def sample_activations() -> Tensor:
    """Sample 2D activation tensor representing linear layer inputs."""
    return torch.randn(4, 256)


@pytest.fixture
def sample_weights() -> Tensor:
    """Sample 2D weight tensor representing linear layer weights."""
    return torch.randn(8, 256)


@pytest.fixture
def sample_conv_activations() -> Tensor:
    """Sample 4D activation tensor representing conv2d inputs."""
    return torch.randn(4, 64, 28, 28)


@pytest.fixture
def sample_conv_weights() -> Tensor:
    """Sample 4D weight tensor representing conv2d kernels."""
    return torch.randn(32, 64, 3, 3)


@pytest.fixture
def sample_linear_activations(sample_activations: Tensor) -> Tensor:
    """Alias for 2D activations to keep test naming consistent."""
    return sample_activations


@pytest.fixture
def sample_linear_weights(sample_weights: Tensor) -> Tensor:
    """Alias for 2D weights to keep test naming consistent."""
    return sample_weights


# ---------------------------------------------------------------------------
# Helper logic shared by multiple test modules
# ---------------------------------------------------------------------------

SUPPORTED_LINEAR_ACTS = {"ai8pt", "ai8ptk", "abf16", "af16", "none"}
SUPPORTED_CONV_ACTS = {"ai8pt", "ai8pc", "abf16", "af16", "none"}

SUPPORTED_LINEAR_WEIGHTS = {
    "wpt",
    "wpc",
    "wpg",
    "wpg64",
    "wpg128",
    "wpg256",
    "wbf16",
    "wf16",
    "none",
}
SUPPORTED_CONV_WEIGHTS = {"wpt", "wpc", "wbf16", "wf16", "none"}


def split_quantizer_name(name: str) -> Tuple[str, str]:
    """Split registry key into activation and weight scheme names."""
    if "_" not in name:
        return name, "none"
    act_name, weight_name = name.split("_", 1)
    return act_name, weight_name


def supports_linear_activation(act_name: str) -> bool:
    """Return True if activation quantizer accepts 2D tensors."""
    return act_name in SUPPORTED_LINEAR_ACTS


def supports_conv_activation(act_name: str) -> bool:
    """Return True if activation quantizer accepts 4D conv tensors."""
    return act_name in SUPPORTED_CONV_ACTS


def supports_linear_weights(weight_name: str) -> bool:
    """Return True if weight quantizer accepts 2D tensors."""
    return weight_name in SUPPORTED_LINEAR_WEIGHTS


def supports_conv_weights(weight_name: str) -> bool:
    """Return True if weight quantizer accepts 4D conv tensors."""
    return weight_name in SUPPORTED_CONV_WEIGHTS


def expected_linear_scale_shape(weight_name: str, w: Tensor) -> torch.Size:
    """Expected scale tensor shape for 2D weights under each scheme."""
    if weight_name in {"wpt", "wbf16", "wf16"}:
        return torch.Size([])
    if weight_name == "none":
        return torch.Size([1])
    if weight_name == "wpc":
        return torch.Size([w.shape[0], 1])
    if weight_name.startswith("wpg"):
        if weight_name == "wpg":
            group_size = 128
        else:
            group_size = int(weight_name.removeprefix("wpg"))
        num_groups = ceil(w.shape[1] / group_size)
        return torch.Size([w.shape[0], num_groups])
    raise AssertionError(f"Unhandled weight quantizer '{weight_name}'")


_QUANTIZER_ITEMS = list(QUANTIZER_REGISTRY.items())

LINEAR_QUANTIZERS = []
CONV_QUANTIZERS = []
for name, cls in _QUANTIZER_ITEMS:
    act_name, weight_name = split_quantizer_name(name)
    if supports_linear_activation(act_name) and supports_linear_weights(weight_name):
        LINEAR_QUANTIZERS.append((name, cls))
    if supports_conv_activation(act_name) and supports_conv_weights(weight_name):
        CONV_QUANTIZERS.append((name, cls))

LINEAR_WEIGHT_QUANTIZERS = [
    (name, fn)
    for name, fn in WEIGHT_QUANT_REGISTRY.items()
    if supports_linear_weights(name)
]

CONV_WEIGHT_QUANTIZERS = [
    (name, fn)
    for name, fn in WEIGHT_QUANT_REGISTRY.items()
    if supports_conv_weights(name)
]

LINEAR_RANGE_WEIGHT_QUANTIZERS = [
    (name, fn)
    for name, fn in WEIGHT_QUANT_REGISTRY.items()
    if name in {"wpt", "wpc"} or name.startswith("wpg")
]

_ACT_ITEMS = list(ACT_QUANT_REGISTRY.items())

LINEAR_ACT_QUANTIZERS = []
CONV_ACT_QUANTIZERS = []
for name, fn in _ACT_ITEMS:
    if supports_linear_activation(name):
        LINEAR_ACT_QUANTIZERS.append((name, fn))
    if supports_conv_activation(name):
        CONV_ACT_QUANTIZERS.append((name, fn))


@pytest.fixture(params=LINEAR_QUANTIZERS, ids=[name for name, _ in LINEAR_QUANTIZERS])
def linear_quantizer(request):
    """Quantizers that support 2D linear tensors."""
    return request.param


@pytest.fixture(params=CONV_QUANTIZERS, ids=[name for name, _ in CONV_QUANTIZERS])
def conv_quantizer(request):
    """Quantizers that support 4D conv tensors."""
    return request.param


@pytest.fixture(params=LINEAR_WEIGHT_QUANTIZERS, ids=[name for name, _ in LINEAR_WEIGHT_QUANTIZERS])
def linear_weight_quantizer(request):
    """Weight quantizers valid for 2D weights."""
    return request.param


@pytest.fixture(params=CONV_WEIGHT_QUANTIZERS, ids=[name for name, _ in CONV_WEIGHT_QUANTIZERS])
def conv_weight_quantizer(request):
    """Weight quantizers valid for 4D conv weights."""
    return request.param


@pytest.fixture(
    params=LINEAR_RANGE_WEIGHT_QUANTIZERS,
    ids=[name for name, _ in LINEAR_RANGE_WEIGHT_QUANTIZERS],
)
def linear_range_weight_quantizer(request):
    """Weight quantizers that enforce a [-1, 1] range."""
    return request.param


@pytest.fixture(params=LINEAR_ACT_QUANTIZERS, ids=[name for name, _ in LINEAR_ACT_QUANTIZERS])
def linear_act_quantizer(request):
    """Activation quantizers valid for 2D tensors."""
    return request.param


@pytest.fixture(params=CONV_ACT_QUANTIZERS, ids=[name for name, _ in CONV_ACT_QUANTIZERS])
def conv_act_quantizer(request):
    """Activation quantizers valid for 4D tensors."""
    return request.param


__all__ = [
    "sample_activations",
    "sample_weights",
    "sample_conv_activations",
    "sample_conv_weights",
    "sample_linear_activations",
    "sample_linear_weights",
    "split_quantizer_name",
    "supports_linear_activation",
    "supports_conv_activation",
    "supports_linear_weights",
    "supports_conv_weights",
    "expected_linear_scale_shape",
    "linear_quantizer",
    "conv_quantizer",
    "linear_weight_quantizer",
    "conv_weight_quantizer",
    "linear_range_weight_quantizer",
    "linear_act_quantizer",
    "conv_act_quantizer",
    "ACT_QUANT_REGISTRY",
    "WEIGHT_QUANT_REGISTRY",
    "QUANTIZER_REGISTRY",
]


"""
Pytest configuration and fixtures for BitLab tests
"""
import pytest
import torch
import sys
import os

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

@pytest.fixture
def device():
    """Fixture to provide device for tests"""
    return torch.device("cpu")


@pytest.fixture
def sample_input_2d():
    """Fixture to provide 2D sample input tensor"""
    return torch.randn(3, 10)


@pytest.fixture
def sample_input_3d():
    """Fixture to provide 3D sample input tensor"""
    return torch.randn(2, 3, 10)


@pytest.fixture
def default_quant_config():
    """Fixture to provide default quantization config"""
    from bitcore.config import BitQuantConfig
    return BitQuantConfig()


@pytest.fixture
def per_channel_quant_config():
    """Fixture to provide per-channel quantization config"""
    from bitcore.config import BitQuantConfig
    return BitQuantConfig(
        weight_granularity="per_channel",
        activation_granularity="per_channel"
    )


@pytest.fixture
def bitlinear_layer():
    """Fixture to provide a BitLinear layer for testing"""
    from bitlayers.bitlinear import BitLinear
    return BitLinear(in_features=10, out_features=5)


@pytest.fixture
def bitlinear_layer_no_bias():
    """Fixture to provide a BitLinear layer without bias for testing"""
    from bitlayers.bitlinear import BitLinear
    return BitLinear(in_features=10, out_features=5, bias=False)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Fixture to reset random seed before each test"""
    torch.manual_seed(42)
    yield
    torch.manual_seed(42)

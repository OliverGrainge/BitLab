"""Unit tests for activation quantization functions."""
import pytest
import torch
from bitlab.bitquantizer.registry import ACT_QUANT_REGISTRY

ACT_QUANT_FUNCTIONS = list(ACT_QUANT_REGISTRY.values())
ACT_QUANT_NAMES = list(ACT_QUANT_REGISTRY.keys())


# Fixtures
@pytest.fixture
def sample_activations():
    """Fixture providing sample activation tensors."""
    return torch.randn(4, 256)


@pytest.fixture(params=zip(ACT_QUANT_NAMES, ACT_QUANT_FUNCTIONS), ids=ACT_QUANT_NAMES)
def act_quantizer(request):
    """Fixture providing activation quantizer name and function."""
    return request.param


@pytest.mark.unit
class TestActivationQuantizationOutput:
    """Unit tests for activation quantization output format."""
    
    def test_output_shape(self, act_quantizer, sample_activations):
        """Test that activation quantization returns correct output shapes."""
        act_quant_name, act_quant_fn = act_quantizer
        x = sample_activations
        
        scale, quantized = act_quant_fn(x)
        
        assert isinstance(scale, torch.Tensor), f"Scale should be torch.Tensor for {act_quant_name}"
        assert quantized.shape == x.shape, f"Quantized should have same shape as input for {act_quant_name}"
    
    def test_output_type(self, act_quantizer, sample_activations):
        """Test that activation quantization returns torch.Tensor."""
        act_quant_name, act_quant_fn = act_quantizer
        x = sample_activations
        
        scale, quantized = act_quant_fn(x)
        
        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)


@pytest.mark.unit
class TestActivationQuantizationRange:
    """Unit tests for activation quantization value ranges."""
    
    def test_quantized_values_in_range(self, act_quantizer, sample_activations):
        """Test that quantized values are in [-127, 127] range."""
        act_quant_name, act_quant_fn = act_quantizer
        x = sample_activations
        
        _, quantized = act_quant_fn(x)
        
        assert (quantized >= -127).all(), f"Quantized values should be >= -127 for {act_quant_name}"
        assert (quantized <= 127).all(), f"Quantized values should be <= 127 for {act_quant_name}"


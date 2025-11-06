"""Unit tests for activation quantization functions."""
import pytest
import torch
from bitlab.bitquantizer.registry import ACT_QUANT_REGISTRY

ACT_QUANT_FUNCTIONS = list(ACT_QUANT_REGISTRY.values())
ACT_QUANT_NAMES = list(ACT_QUANT_REGISTRY.keys())


# Fixtures
@pytest.fixture
def sample_activations():
    """Fixture providing sample activation tensors for linear layers (2D)."""
    return torch.randn(4, 256)


@pytest.fixture
def sample_conv_activations():
    """Fixture providing sample activation tensors for conv layers (4D)."""
    return torch.randn(4, 64, 28, 28)


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


@pytest.mark.unit
class TestConvActivationQuantizationOutput:
    """Unit tests for conv2d activation quantization (4D tensors) output format."""
    
    def test_output_shape(self, act_quantizer, sample_conv_activations):
        """Test that activation quantization returns correct output shapes for conv2d."""
        act_quant_name, act_quant_fn = act_quantizer
        x = sample_conv_activations
        
        scale, quantized = act_quant_fn(x)
        
        assert isinstance(scale, torch.Tensor), f"Scale should be torch.Tensor for {act_quant_name}"
        assert quantized.shape == x.shape, f"Quantized should have same shape as input for {act_quant_name}"
    
    def test_output_type(self, act_quantizer, sample_conv_activations):
        """Test that activation quantization returns torch.Tensor for conv2d."""
        act_quant_name, act_quant_fn = act_quantizer
        x = sample_conv_activations
        
        scale, quantized = act_quant_fn(x)
        
        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)
    
    def test_scale_shape_for_per_channel(self, sample_conv_activations):
        """Test that ai8pc produces correct scale shape for conv2d (per-channel over spatial)."""
        from bitlab.bitquantizer.act import quantize_act_ai8pc
        x = sample_conv_activations  # [batch, channels, height, width]
        
        scale, quantized = quantize_act_ai8pc(x)
        
        # For conv2d, ai8pc should produce per-channel scales over spatial dimensions
        expected_scale_shape = (x.shape[0], x.shape[1], 1, 1)
        assert scale.shape == expected_scale_shape, f"Scale shape should be {expected_scale_shape} for conv2d, got {scale.shape}"
        assert quantized.shape == x.shape


@pytest.mark.unit
class TestConvActivationQuantizationRange:
    """Unit tests for conv2d activation quantization value ranges."""
    
    def test_quantized_values_in_range(self, act_quantizer, sample_conv_activations):
        """Test that quantized values are in [-127, 127] range for conv2d."""
        act_quant_name, act_quant_fn = act_quantizer
        x = sample_conv_activations
        
        _, quantized = act_quant_fn(x)
        
        assert (quantized >= -127).all(), f"Quantized values should be >= -127 for {act_quant_name}"
        assert (quantized <= 127).all(), f"Quantized values should be <= 127 for {act_quant_name}"


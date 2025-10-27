import pytest
import torch
import torch.nn as nn
import math
from bitlayers.bitlinear import BitLinear
from bitcore.config import BitQuantConfig


class TestBitLinear:
    """Test cases for BitLinear class"""
    
    @pytest.mark.unit
    def test_init_basic(self):
        """Test basic initialization"""
        layer = BitLinear(in_features=10, out_features=5)
        
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.use_bias is True
        assert layer.weight.shape == (5, 10)
        assert layer.bias is not None
        assert layer.bias.shape == (5,)
    
    @pytest.mark.unit
    def test_init_without_bias(self):
        """Test initialization without bias"""
        layer = BitLinear(in_features=10, out_features=5, bias=False)
        
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.use_bias is False
        assert layer.weight.shape == (5, 10)
        assert layer.bias is None
    
    @pytest.mark.unit
    def test_init_with_custom_quant_config(self):
        """Test initialization with custom quantization config"""
        custom_config = BitQuantConfig(
            weight_granularity="per_channel",
            activation_granularity="per_channel"
        )
        layer = BitLinear(in_features=10, out_features=5, quant_config=custom_config)
        
        assert layer.quant_config == custom_config
        assert layer.quant_config.weight_granularity == "per_channel"
        assert layer.quant_config.activation_granularity == "per_channel"
    
    
    @pytest.mark.unit
    def test_weight_initialization_kaiming_uniform(self):
        """Test Kaiming uniform weight initialization"""
        layer = BitLinear(in_features=10, out_features=5, init_method='kaiming_uniform')
        
        # Check that weights are within expected bounds
        bound = math.sqrt(3.0) * math.sqrt(2.0 / 10)
        assert torch.all(layer.weight >= -bound)
        assert torch.all(layer.weight <= bound)
    
    
    @pytest.mark.unit
    def test_invalid_initialization_method(self):
        """Test that invalid initialization method raises ValueError"""
        with pytest.raises(ValueError, match="Unknown initialization method"):
            BitLinear(in_features=10, out_features=5, init_method='invalid_method')
    
    @pytest.mark.unit
    def test_quantization_params_per_tensor(self):
        """Test quantization parameter creation for per_tensor granularity"""
        config = BitQuantConfig(
            weight_granularity="per_tensor",
            activation_granularity="per_tensor"
        )
        layer = BitLinear(in_features=10, out_features=5, quant_config=config)
        
        # Initially, quantization parameters should not exist
        assert not hasattr(layer, 'qweight_scale')
        assert not hasattr(layer, 'qweight')
        
        # Enter eval mode to create quantization parameters
        layer.eval()
        
        # Check weight quantization parameters
        assert hasattr(layer, 'qweight_scale')
        assert layer.qweight_scale.shape == torch.Size([])  # scalar tensor for per_tensor
        assert hasattr(layer, 'qweight')
        assert layer.qweight.shape == (5, 10)  # out_features, in_features
    
    @pytest.mark.unit
    def test_quantization_params_per_channel(self):
        """Test quantization parameter creation for per_channel granularity"""
        config = BitQuantConfig(
            weight_granularity="per_channel",
            activation_granularity="per_channel"
        )
        layer = BitLinear(in_features=10, out_features=5, quant_config=config)
        
        # Initially, quantization parameters should not exist
        assert not hasattr(layer, 'qweight_scale')
        assert not hasattr(layer, 'qweight')
        
        # Enter eval mode to create quantization parameters
        layer.eval()
        
        # Check weight quantization parameters (per output channel)
        assert hasattr(layer, 'qweight_scale')
        assert layer.qweight_scale.shape == (5,)  # out_features
        assert hasattr(layer, 'qweight')
        assert layer.qweight.shape == (5, 10)  # out_features, in_features
    
    @pytest.mark.unit
    def test_forward_pass_training_mode(self):
        """Test forward pass in training mode"""
        layer = BitLinear(in_features=10, out_features=5)
        layer.train()
        
        input_tensor = torch.randn(3, 10)
        output = layer(input_tensor)
        
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.unit
    def test_forward_pass_eval_mode(self):
        """Test forward pass in evaluation mode"""
        layer = BitLinear(in_features=10, out_features=5)
        layer.eval()
        
        input_tensor = torch.randn(3, 10)
        output = layer(input_tensor)
        
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.unit
    def test_forward_pass_without_bias(self):
        """Test forward pass without bias"""
        layer = BitLinear(in_features=10, out_features=5, bias=False)
        layer.eval()
        
        input_tensor = torch.randn(3, 10)
        output = layer(input_tensor)
        
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.unit
    def test_forward_pass_batch_dimensions(self):
        """Test forward pass with different batch dimensions"""
        layer = BitLinear(in_features=10, out_features=5)
        layer.eval()
        
        # Test 2D input
        input_2d = torch.randn(3, 10)
        output_2d = layer(input_2d)
        assert output_2d.shape == (3, 5)
        
        # Test 3D input
        input_3d = torch.randn(2, 3, 10)
        output_3d = layer(input_3d)
        assert output_3d.shape == (2, 3, 5)
    
    @pytest.mark.unit
    def test_mode_switching_methods(self):
        """Test that mode switching methods are callable"""
        layer = BitLinear(in_features=10, out_features=5)
        
        # These should not raise exceptions
        layer._on_enter_training_mode()
        layer._on_enter_eval_mode()
    
    @pytest.mark.unit
    def test_parameters_are_trainable(self):
        """Test that all parameters are trainable"""
        layer = BitLinear(in_features=10, out_features=5)
        
        # Check that only the main parameters are registered initially
        param_names = [name for name, _ in layer.named_parameters()]
        assert 'weight' in param_names
        assert 'bias' in param_names
        # Quantization parameters are not trainable parameters, they're buffers created in eval mode
        assert 'qweight_scale' not in param_names
        assert 'qx_scale' not in param_names
        
        # Check that quantization parameters are created as buffers in eval mode
        layer.eval()
        buffer_names = [name for name, _ in layer.named_buffers()]
        assert 'qweight_scale' in buffer_names
        assert 'qweight' in buffer_names
    
    @pytest.mark.unit
    def test_parameters_are_trainable_no_bias(self):
        """Test that parameters are trainable when bias=False"""
        layer = BitLinear(in_features=10, out_features=5, bias=False)
        
        # Check that only the main parameters are registered initially
        param_names = [name for name, _ in layer.named_parameters()]
        assert 'weight' in param_names
        assert 'bias' not in param_names
        # Quantization parameters are not trainable parameters, they're buffers created in eval mode
        assert 'qweight_scale' not in param_names
        assert 'qx_scale' not in param_names
        
        # Check that quantization parameters are created as buffers in eval mode
        layer.eval()
        buffer_names = [name for name, _ in layer.named_buffers()]
        assert 'qweight_scale' in buffer_names
        assert 'qweight' in buffer_names

    @pytest.mark.unit
    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer"""
        layer = BitLinear(in_features=10, out_features=5)
        layer.train()
        
        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert input_tensor.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

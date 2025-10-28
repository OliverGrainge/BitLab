import pytest
import torch
import torch.nn as nn
import math
from bitcore.ops import bitlinear
from bitcore.config import BitQuantConfig


class TestOpsBitLinearAsptWpt:
    """Test cases for BitLinear class"""
    
    @pytest.mark.unit
    def test_train_forward_basic(self):
        quant_config = BitQuantConfig(
            weight_granularity="per_tensor",
            activation_granularity="per_tensor",
            activation_dtype="float32"
        )
        x = torch.randn(10, 9)
        w = torch.randn(15, 9)
        bias = torch.randn(15)
        y = bitlinear.train_forward(x, w, bias, quant_config)
        assert y.shape == (10, 15)

        
    @pytest.mark.unit
    def test_eval_forward_basic(self): 
        quant_config = BitQuantConfig(
            weight_granularity="per_tensor",
            activation_granularity="per_tensor",
            activation_dtype="float32"
        )
        x = torch.randn(10, 9)
        w = torch.randn(15, 9)
        bias = torch.randn(15)
        qweight_scale, qweight = bitlinear.quantize_weights(w, quant_config)
        y = bitlinear.eval_forward(x, qweight_scale, qweight, bias, quant_config)
        assert y.shape == (10, 15)


    @pytest.mark.unit 
    def test_eval_train_equivalence(self): 
        quant_config = BitQuantConfig(
            weight_granularity="per_tensor",
            activation_granularity="per_tensor",
            activation_dtype="float32"
        )
        torch.manual_seed(42)  # Ensure deterministic random values
        x = torch.randn(2, 2)
        w = torch.randn(2, 2)
        bias = torch.randn(2)
        qweight_scale, qweight = bitlinear.quantize_weights(w, quant_config)
        y_eval = bitlinear.eval_forward(x, qweight_scale, qweight, bias, quant_config)
        y_train = bitlinear.train_forward(x, w, bias, quant_config)
        # Check that the outputs are close but allow for small quantization error,
        # and supply a useful debug message if the check fails.
        assert torch.allclose(y_eval, y_train, atol=1e-6), (
            f"eval_forward and train_forward outputs are not sufficiently close.\n"
            f"Max abs diff: {(y_eval - y_train).abs().max().item()}\n"
            f"y_eval: {y_eval}\n"
            f"y_train: {y_train}"
        )




    
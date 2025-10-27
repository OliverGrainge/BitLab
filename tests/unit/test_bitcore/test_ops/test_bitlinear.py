import pytest
import torch
import torch.nn as nn
import math
from bitcore.ops import bitlinear
from bitcore.config import BitQuantConfig


class TestOpsBitLinearAi8ptWpt:
    """Test cases for BitLinear class"""
    
    @pytest.mark.unit
    def test_train_forward_basic(self):
        quant_config = BitQuantConfig(
            weight_granularity="per_tensor",
            activation_granularity="per_tensor",
            activation_dtype="int8"
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
            activation_dtype="int8"
        )
        x = torch.randn(10, 9)
        w = torch.randn(15, 9)
        bias = torch.randn(15)
        qweight_scale, qweight = bitlinear.quantize_weights(w, quant_config)
        y = bitlinear.eval_forward(x, qweight_scale, qweight, bias, quant_config)
        assert y.shape == (10, 15)



    
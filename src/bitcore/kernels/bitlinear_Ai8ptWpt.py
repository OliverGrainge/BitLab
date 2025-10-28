# bitcore/kernels/bitlinear_kernel_i8_pt_pt.py
import torch
from bitcore.kernels.base import OptimizedKernelBase
from bitcore.config import BitQuantConfig
from bitcore.kernels.registry import KernelRegistry

# Import the optimized C++ bindings
try:
    from bitcore.kernels.bindings import bitlinear_int8_pt_pt_cpp
    CPP_MODULE = bitlinear_int8_pt_pt_cpp
except ImportError:
    CPP_MODULE = None


@KernelRegistry.register_kernel(
    name="BitLinearAi8PtWptCpp",
    priority=100,
    activation_dtype="int8",
    activation_granularity="per_tensor",
    weight_granularity="per_tensor"
)
class BitLinearAi8PtWptCpp(OptimizedKernelBase):
    """Optimized kernel for INT8 per-tensor quantization"""
    
    def __init__(self, config: BitQuantConfig = None):
        if config is None:
            config = BitQuantConfig(
                activation_dtype="int8",
                activation_granularity="per_tensor",
                weight_granularity="per_tensor"
            )
        super().__init__(cpp_module=CPP_MODULE, quant_config=config)
    
    @classmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Check if compatible with INT8 per-tensor config"""
        return (
            config.activation_dtype == "int8" and
            config.activation_granularity == "per_tensor" and
            config.weight_granularity == "per_tensor"
        )
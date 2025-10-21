# bitcore/kernels/registry.py
from typing import Dict, Type, Optional, Callable
from bitcore.config import BitQuantConfig
from bitcore.kernels.base import BitKernelBase, ReferenceKernel

class KernelRegistry:
    """Registry for managing bitlinear kernels with decorator support"""
    
    _kernels: Dict[str, Type[BitKernelBase]] = {}
    _dispatch_rules: Dict[str, Callable[[BitQuantConfig], bool]] = {}
    _default_kernel: Type[BitKernelBase] = ReferenceKernel
    
    @classmethod
    def register(cls, config_matcher: Callable[[BitQuantConfig], bool], name: str = None):
        """
        Single decorator to register kernels with config matching
        
        Usage:
            @KernelRegistry.register(lambda config: config.activation_dtype == "int8")
            class Int8Kernel(BitKernelBase):
                pass
                
            @KernelRegistry.register(
                lambda config: (
                    config.activation_dtype == "int8" and 
                    config.activation_granularity == "per_tensor"
                ),
                name="custom_int8"
            )
            class CustomInt8Kernel(BitKernelBase):
                pass
        """
        def decorator(kernel_class: Type[BitKernelBase]):
            # Use class name if no name provided
            kernel_name = name or kernel_class.__name__
            
            # Register the kernel
            cls._kernels[kernel_name] = kernel_class
            
            # Register dispatch rule
            cls._dispatch_rules[kernel_name] = config_matcher
            
            return kernel_class
        
        return decorator
    
    @classmethod
    def get_kernel_for_config(cls, quant_config: BitQuantConfig) -> BitKernelBase:
        """Get the best kernel for the given config"""
        
        # Check dispatch rules in order of registration
        for name, matcher in cls._dispatch_rules.items():
            if matcher(quant_config):
                kernel_class = cls._kernels[name]
                return kernel_class(quant_config)
        
        # Fallback to default kernel
        return cls._default_kernel(quant_config)
    
    @classmethod
    def get_kernel_by_name(cls, name: str, quant_config: BitQuantConfig) -> BitKernelBase:
        """Get a specific kernel by name"""
        if name not in cls._kernels:
            raise ValueError(f"Unknown kernel: {name}")
        return cls._kernels[name](quant_config)
    
    @classmethod
    def list_available_kernels(cls):
        """List all available kernels"""
        return list(cls._kernels.keys())
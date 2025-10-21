# bitcore/kernels/registry.py
from typing import Dict, Type, Optional, Callable, List, Tuple
from bitcore.config import BitQuantConfig
from bitcore.kernels.base import BitKernelBase, ReferenceKernel


class KernelRegistry:
    """Registry for managing bitlinear kernels with decorator support"""
    
    # Store (priority, name, class, matcher) tuples
    _entries: List[Tuple[int, str, Type[BitKernelBase], Callable]] = []
    _default_kernel: Type[BitKernelBase] = ReferenceKernel
    
    @classmethod
    def register(
        cls, 
        config_matcher: Callable[[BitQuantConfig], bool],
        name: str = None,
        priority: int = 0
    ):
        """
        Register a kernel with config matching.
        
        Higher priority kernels are checked first.
        
        Usage:
            @KernelRegistry.register(
                lambda config: config.activation_dtype == "int8",
                priority=100
            )
            class Int8Kernel(BitKernelBase):
                pass
        """
        def decorator(kernel_class: Type[BitKernelBase]):
            kernel_name = name or kernel_class.__name__
            
            # Add entry with priority
            cls._entries.append((priority, kernel_name, kernel_class, config_matcher))
            
            # Sort by priority (highest first)
            cls._entries.sort(key=lambda x: x[0], reverse=True)
            
            return kernel_class
        
        return decorator
    
    @classmethod
    def get_kernel_from_config(cls, quant_config: BitQuantConfig) -> BitKernelBase:
        """Get the best kernel for the given config"""
        
        # Check entries in priority order
        for priority, name, kernel_class, matcher in cls._entries:
            if matcher(quant_config):
                return kernel_class(quant_config)
        
        # Fallback to default kernel
        return cls._default_kernel(quant_config)
    
    @classmethod
    def get_kernel_by_name(cls, name: str, quant_config: BitQuantConfig) -> BitKernelBase:
        """Get a specific kernel by name"""
        for _, kernel_name, kernel_class, _ in cls._entries:
            if kernel_name == name:
                return kernel_class(quant_config)
        raise ValueError(f"Unknown kernel: {name}")
    
    @classmethod
    def list_available_kernels(cls):
        """List all available kernels"""
        return [name for _, name, _, _ in cls._entries]
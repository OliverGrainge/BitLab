from typing import Type, Callable, List, Optional, Literal
from dataclasses import dataclass
from bitcore.config import BitQuantConfig
from bitcore.kernels.base import BitKernelBase, ReferenceKernel
import logging

logger = logging.getLogger(__name__)


@dataclass
class KernelEntry:
    priority: int
    name: str
    kernel_class: Type[BitKernelBase]
    matcher: Callable[[BitQuantConfig], bool]


class NoSuitableKernelError(Exception):
    """Raised when no suitable kernel is found for a given configuration"""
    pass


class KernelRegistry:
    """Registry for managing bitlinear kernels"""
    
    _entries: List[KernelEntry] = []
    _default_kernel: Type[BitKernelBase] = ReferenceKernel
    
    @classmethod
    def register(cls, matcher: Callable[[BitQuantConfig], bool], 
                 name: str = None, priority: int = 0):
        """Register a kernel with priority-based dispatch"""
        def decorator(kernel_class: Type[BitKernelBase]):
            entry = KernelEntry(
                priority=priority,
                name=name or kernel_class.__name__,
                kernel_class=kernel_class,
                matcher=matcher
            )
            cls._entries.append(entry)
            cls._entries.sort(key=lambda e: e.priority, reverse=True)
            return kernel_class
        return decorator
    
    @classmethod
    def register_kernel(cls, name: str, priority: int = 0, 
                       activation_dtype: Optional[Literal["float32", "int8"]] = None,
                       activation_granularity: Optional[Literal["per_tensor", "per_channel"]] = None,
                       weight_granularity: Optional[Literal["per_tensor", "per_channel"]] = None):
        """Declarative kernel registration with cleaner syntax"""
        def matcher(config: BitQuantConfig) -> bool:
            if activation_dtype is not None and config.activation_dtype != activation_dtype:
                return False
            if activation_granularity is not None and config.activation_granularity != activation_granularity:
                return False
            if weight_granularity is not None and config.weight_granularity != weight_granularity:
                return False
            return True
        
        def decorator(kernel_class: Type[BitKernelBase]):
            entry = KernelEntry(
                priority=priority,
                name=name,
                kernel_class=kernel_class,
                matcher=matcher
            )
            cls._entries.append(entry)
            cls._entries.sort(key=lambda e: e.priority, reverse=True)
            return kernel_class
        return decorator
    
    @classmethod
    def get_kernel_from_config(cls, config: BitQuantConfig) -> BitKernelBase:
        """Get best kernel for config with proper error handling"""
        for entry in cls._entries:
            if entry.matcher(config):
                return entry.kernel_class(config)
        
        # Fall back to default kernel with warning
        logger.warning(f"No optimized kernel found for config {config}, using reference kernel")
        return cls._default_kernel(config)
    
    @classmethod
    def get_kernel_by_name(cls, name: str, config: BitQuantConfig) -> BitKernelBase:
        """Get specific kernel by name"""
        for entry in cls._entries:
            if entry.name == name:
                return entry.kernel_class(config)
        raise ValueError(f"Unknown kernel: {name}")
    
    @classmethod
    def list_kernels(cls) -> List[str]:
        """List all registered kernels in priority order"""
        return [e.name for e in cls._entries]
    
    @classmethod
    def get_optimal_kernel(cls, config: BitQuantConfig) -> BitKernelBase:
        """Get the best kernel for the given configuration with proper error handling"""
        try:
            return cls.get_kernel_from_config(config)
        except Exception as e:
            logger.error(f"Error getting kernel for config {config}: {e}")
            logger.warning("Falling back to reference kernel")
            return cls._default_kernel(config)
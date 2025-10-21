from typing import Type, Callable, List
from dataclasses import dataclass
from bitcore.config import BitQuantConfig
from bitcore.kernels.base import BitKernelBase, ReferenceKernel


@dataclass
class KernelEntry:
    priority: int
    name: str
    kernel_class: Type[BitKernelBase]
    matcher: Callable[[BitQuantConfig], bool]


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
    def get_kernel_from_config(cls, config: BitQuantConfig) -> BitKernelBase:
        """Get best kernel for config"""
        for entry in cls._entries:
            if entry.matcher(config):
                return entry.kernel_class(config)
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
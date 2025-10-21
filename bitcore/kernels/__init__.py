from .registry import KernelRegistry
from .base import BitKernelBase, ReferenceKernel

def _auto_discover_kernels():
    """Automatically discover and register all kernels in this package"""
    import pkgutil
    import importlib
    import logging
    
    logger = logging.getLogger(__name__)
    
    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        if not ispkg and modname.startswith('bitlinear_kernel_'):
            try:
                importlib.import_module(f'.{modname}', __package__)
                logger.debug(f"Auto-discovered kernel: {modname}")
            except Exception as e:
                logger.warning(f"Failed to import kernel {modname}: {e}")

# Auto-discover kernels on import
_auto_discover_kernels()

__all__ = ["KernelRegistry", "BitKernelBase", "ReferenceKernel"]

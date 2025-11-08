import torch.nn as nn
import torch


class Module(nn.Module):
    """Base class for BitLab modules with an opt-in deployment hook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def deploy(self) -> nn.Module:
        """Invoke `_deploy` on child modules and return self for chaining."""
        for module in self.modules():
            if hasattr(module, "_deploy"):
                module._deploy()
        return self

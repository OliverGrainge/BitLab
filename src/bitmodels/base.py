import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BitModelBase(nn.Module, ABC):
    """
    Base class for all BitLab models.
    
    This class provides a common interface and structure for all models
    in the BitLab framework. All model implementations should inherit
    from this base class.
    """
    
    def __init__(self, config: Any, quant_config: Optional[Dict] = None):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration object
            quant_config: Optional quantization configuration
        """
        super().__init__()
        self.config = config
        self.quant_config = quant_config
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_config(self) -> Any:
        """
        Get the model configuration.
        
        Returns:
            Model configuration object
        """
        return self.config
    
    def get_quant_config(self) -> Optional[Dict]:
        """
        Get the quantization configuration.
        
        Returns:
            Quantization configuration dict or None
        """
        return self.quant_config
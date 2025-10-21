import torch.nn as nn 
from bitcore import BitQuantConfig

class BitLayerBase(nn.Module): 
    def __init__(self, quant_config: BitQuantConfig = None):
        super().__init__()
        self.quant_config = quant_config if quant_config is not None else BitQuantConfig()

    def _init_quantization_params(self):
        """
        Initialize quantization-specific parameters based on quant_config.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _init_quantization_params")

    def forward(self, *args, **kwargs):
        if self.training: 
            return self._train_forward(*args, **kwargs)
        else: 
            return self._eval_forward(*args, **kwargs)

    def _train_forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement _train_forward")

    def _eval_forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement _eval_forward")

    def train(self, mode=True):
        """Override train() to perform custom transformations when entering training mode"""
        super().train(mode)
        if mode:
            self._on_enter_training_mode()
        else:
            # Optionally handle the case when mode=False
            self._on_enter_eval_mode()
        return self
    
    def eval(self):
        """Override eval() to perform custom transformations when entering evaluation mode"""
        super().eval()
        self._on_enter_eval_mode()
        return self


    def _on_enter_training_mode(self):
        """Called when entering training mode - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _on_enter_training_mode")

    def _on_enter_eval_mode(self):
        """Called when entering evaluation mode - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _on_enter_eval_mode")
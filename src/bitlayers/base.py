import torch.nn as nn 
from bitcore import BitQuantConfig

class BitLayerBase(nn.Module): 
    def __init__(self, quant_config: BitQuantConfig = None):
        super().__init__()
        self.quant_config = quant_config if quant_config is not None else BitQuantConfig()
        self._is_deployed = False  # Track deployment state

    def _init_quantization_params(self):
        """
        Initialize quantization-specific parameters based on quant_config.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _init_quantization_params")

    def forward(self, *args, **kwargs):
        if self._is_deployed or not self.training: 
            return self._eval_forward(*args, **kwargs)
        else: 
            return self._train_forward(*args, **kwargs)

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

    def _compute_reg(self): 
        """Compute regularization loss"""
        return 0.0

    def deploy(self):
            """
            Permanently quantize the layer and remove latent weights.
            After deployment, the layer becomes inference-only and cannot be trained.
            
            This method should be overridden by subclasses to implement specific
            quantization and weight removal logic.
            
            Returns:
                self: Returns self for method chaining
            """
            if self._is_deployed:
                raise RuntimeError("Layer has already been deployed")
            
            # Set to eval mode before deployment
            self.eval()
            
            # Call subclass-specific deployment logic
            self._perform_deployment()
            
            # Mark as deployed
            self._is_deployed = True
            
            return self
    
    def _perform_deployment(self):
        """
        Perform the actual deployment operations (quantization, weight removal).
        Must be implemented by subclasses.
        
        Example implementation in subclass:
        - Quantize weights permanently
        - Delete original float weights
        - Delete any training-only parameters
        - Optimize buffers for inference
        """
        raise NotImplementedError("Subclasses must implement _perform_deployment")
    
    @property
    def is_deployed(self):
        """Check if the layer has been deployed"""
        return self._is_deployed

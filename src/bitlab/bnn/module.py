import torch.nn as nn 
import torch 

class Module(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def deploy(self) -> nn.Module: 
        for module in self.modules(): 
            if hasattr(module, '_deploy'): 
                module._deploy()
        return self


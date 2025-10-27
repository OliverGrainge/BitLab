import torch 
from bitcore.config import BitQuantConfig



def compute_weight_scale(weight: torch.Tensor, quant_config: BitQuantConfig): 
    if quant_config.weight_granularity == "per_tensor": 
        return weight.abs().mean() 
    elif quant_config.weight_granularity == "per_channel": 
        return weight.abs().mean(dim=1)
    else:
        raise ValueError(f"Invalid weight granularity: {quant_config.weight_granularity}")


def compute_x_scale(x: torch.Tensor, quant_config: BitQuantConfig): 
    if quant_config.activation_dtype == "float32": 
        return torch.tensor(1.0)
    elif quant_config.activation_dtype == "int8": 
        if quant_config.activation_granularity == "per_tensor": 
            max_abs = x.abs().max() 
            return torch.tensor(max_abs/torch.tensor(127.0))
        elif quant_config.activation_granularity == "per_channel": 
            max_abs = x.abs().amax(dim=1)
            return max_abs/torch.tensor(127.0)
        else: 
            raise ValueError(f"Invalid activation granularity: {quant_config.activation_granularity}")
    else: 
        raise ValueError(f"Invalid activation dtype: {quant_config.activation_dtype}")


def quantize_weight(weight: torch.Tensor, qweight_scale: torch.Tensor, 
                    quant_config: BitQuantConfig, eps: float = 1e-6): 
    # Ensure proper broadcasting for per_channel quantization
    if quant_config.weight_granularity == "per_channel":
        # Reshape scale to (out_features, 1) for broadcasting with (out_features, in_features)
        qweight_scale = qweight_scale.unsqueeze(1)
    
    qweight = weight / (qweight_scale + eps)
    # STE: detach the quantized values and add back the original for gradient flow
    qweight = qweight + (torch.clamp(qweight.round(), -1, 1) - qweight).detach()
    return qweight


def quantize_x(x: torch.Tensor, qx_scale: torch.Tensor, 
               quant_config: BitQuantConfig, eps: float = 1e-6, 
               out_dtype: torch.dtype = None): 
    qx = x / (qx_scale + eps/127.0)
    # STE: detach the quantized values and add back the original for gradient flow
    qx = qx + (qx.round().clip(-128, 127) - qx).detach()
    if out_dtype is not None:
        qx = qx.to(out_dtype)
    return qx



def dequantize_x(qx: torch.Tensor, qx_scale: torch.Tensor, quant_config: BitQuantConfig): 
    return qx * qx_scale 


def dequantize_weight(qweight: torch.Tensor, qweight_scale: torch.Tensor, quant_config: BitQuantConfig): 
    return qweight * qweight_scale 
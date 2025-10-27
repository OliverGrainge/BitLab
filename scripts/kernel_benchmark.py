from bitcore.kernels import KernelRegistry 
from typing import Tuple
import torch 
from bitcore.config import BitQuantConfig
import time


# Separate shapes for different kernel types
linear_shapes = [
    # (batch, in_features, out_features)
    (16, 32, 32), 
    (16, 64, 64),
    (16, 128, 128),
    (16, 256, 256),

]

conv_shapes = [
    # (batch, in_channels, in_height, in_width, out_channels, kernel_height, kernel_width)
    (1, 3, 64, 64, 8, 3, 3),
    (8, 16, 128, 128, 32, 5, 5),
    (16, 32, 256, 256, 64, 3, 3),
    (32, 64, 512, 512, 128, 3, 3),
]



def benchmark_kernel(kernel_name: str, shape: Tuple[int, ...], device: str = "cpu", num_warmup: int = 10, num_iters: int = 100):
    if "bitlinear" in kernel_name.lower(): 
        return _benchmark_linear_kernel(kernel_name, shape, device, num_warmup, num_iters)
    elif "bitconv" in kernel_name.lower():
        raise NotImplementedError("Conv kernel benchmarking is not implemented yet")
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


def _get_inputs(shape: Tuple[int, ...], device: str = "cpu"):
    if len(shape) == 3: 
        batch, in_features, out_features = shape
        x = torch.randn(batch, in_features, dtype=torch.float32).to(device)
        w = torch.randn(out_features, in_features, dtype=torch.float32).to(device)
        return x, w
    elif len(shape) == 7: 
        batch, in_channels, in_height, in_width, out_channels, kernel_height, kernel_width = shape
        x = torch.randn(batch, in_channels, in_height, in_width, dtype=torch.float32).to(device)
        w = torch.randn(out_channels, in_channels, kernel_height, kernel_width, dtype=torch.float32).to(device)
        return x, w
    else:
        raise ValueError(f"Unknown input shape: {shape}")


def _benchmark_linear_kernel(kernel_name: str, shape: Tuple[int, int, int], device: str = "cpu", num_warmup: int = 10, num_iters: int = 100):
    batch, in_features, out_features = shape
    x, w = _get_inputs(shape, device)
    bias = torch.randn(out_features, dtype=torch.float32).to(device)
    kernel = KernelRegistry().get_kernel_from_name(kernel_name)
    qweight_scale, qweight = kernel.quantize_weights(w) 
    
    # Calculate number of operations (matrix multiplication: batch * in_features * out_features)
    num_ops = batch * in_features * out_features
    
    # Warmup
    for _ in range(num_warmup):
        result = kernel.forward(x, qweight_scale, qweight, bias)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        result = kernel.forward(x, qweight_scale, qweight, bias)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_time = (end - start) / num_iters
    return avg_time, num_ops





if __name__ == "__main__": 
    kernel_registry = KernelRegistry()
    kernel_names = kernel_registry.list_kernels()
    print(kernel_names)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}...")
    
    for kernel_name in kernel_names:
        print(f"\n{'='*60}")
        print(f"Kernel: {kernel_name}")
        print(f"{'='*60}")
        
        if "bitlinear" in kernel_name.lower():
            shapes = linear_shapes
            shape_desc = "batch, in_feat, out_feat"
        elif "bitconv" in kernel_name.lower():
            shapes = conv_shapes
            shape_desc = "batch, channels, H, W"
        else:
            continue
        
        print(f"{'Shape (' + shape_desc + ')':40} | Time (ms) | TOPs/s")
        print("-" * 70)
        
        for shape in shapes:
            avg_time, num_ops = benchmark_kernel(kernel_name, shape, device)
            # Calculate TOPs per second: (ops per iteration) / (time per iteration in seconds) / 1e12
            # For linear layers: ops = batch_size * in_features * out_features
            tops_per_sec = (num_ops / avg_time) / 1e12
            print(f"{str(shape):40} | {avg_time*1000:9.4f} | {tops_per_sec:8.4f}")

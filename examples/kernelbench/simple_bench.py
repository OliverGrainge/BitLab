import torch
import time
import sys
import os
import torch.nn.functional as F

# Make sure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from bitlab.bnn.functional import bitlinear, bitlinear_prepare_weights
from bitlab.bnn.functional.bitlinear import KERNELS_AVAILABLE

def time_it(fn, runs=100, warmup=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean_s = sum(times) / len(times)
    return mean_s

def main():
    shapes = [
        (32, 128, 64),
        (64, 256, 128),
    ]

    runs = 100
    warmup = 10

    print(f"Kernels available: {KERNELS_AVAILABLE}")
    print("CPU-only simple benchmark\n")

    for batch, in_features, out_features in shapes:
        x = torch.randn(batch, in_features)
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        # Pack once for bitlinear
        pw, s = bitlinear_prepare_weights(w)

        # Torch linear (functional)
        torch_mean_s = time_it(lambda: F.linear(x, w, b), runs=runs, warmup=warmup)

        # Bitlinear
        bit_mean_s = time_it(lambda: bitlinear(x, pw, s, b), runs=runs, warmup=warmup)

        torch_ms = torch_mean_s * 1000.0
        bit_ms = bit_mean_s * 1000.0
        speedup = torch_ms / bit_ms if bit_ms > 0 else float('inf')

        # Throughput in GOps/s (count MAC as 2 ops)
        ops_per_forward = 2.0 * batch * in_features * out_features
        torch_gops = (ops_per_forward / torch_mean_s) / 1e9
        bit_gops = (ops_per_forward / bit_mean_s) / 1e9

        print(f"Shape: batch={batch}, in={in_features}, out={out_features}")
        print(f"  Torch linear    : {torch_ms:.3f} ms  | {torch_gops:.2f} GOps/s")
        print(f"  Bitlinear       : {bit_ms:.3f} ms  | {bit_gops:.2f} GOps/s")
        print(f"  Speedup (Torch/Bitlinear): {speedup:.2f}x\n")

if __name__ == "__main__":
    main()

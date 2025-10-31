import torch
import time
import sys
import os
import torch.nn.functional as F

# Make sure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bitlab.bnn.functional import bitlinear, bitlinear_prepare_weights
from bitlab.bnn.functional.bitlinear import KERNELS_AVAILABLE


def time_it(fn, runs=100, warmup=10):
    """Time a function with warmup runs."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_s = sum(times) / len(times)
    std_s = (sum((t - mean_s) ** 2 for t in times) / len(times)) ** 0.5
    return mean_s, std_s


def main():
    # Test a range of sizes from small to large
    shapes = [
        # Small
        (32, 128, 64),
        (64, 256, 128),
        # Medium
        (128, 512, 256),
        (256, 1024, 512),
    ]

    runs = 50
    warmup = 5

    print(f"Kernels available: {KERNELS_AVAILABLE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of threads: {torch.get_num_threads()}")
    print("\n" + "=" * 80)
    print("CPU Benchmark - BitLinear vs PyTorch F.linear")
    print("=" * 80 + "\n")

    for batch, in_features, out_features in shapes:
        x = torch.randn(batch, in_features)
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        # Pack once for bitlinear
        pw, s = bitlinear_prepare_weights(w)

        # Time torch linear
        torch_mean, torch_std = time_it(
            lambda: F.linear(x, w, b), runs=runs, warmup=warmup
        )

        # Time bitlinear
        bit_mean, bit_std = time_it(
            lambda: bitlinear(x, pw, s, b), runs=runs, warmup=warmup
        )

        torch_ms = torch_mean * 1000.0
        bit_ms = bit_mean * 1000.0
        speedup = torch_ms / bit_ms if bit_ms > 0 else float("inf")

        # Calculate throughput
        ops_per_forward = 2.0 * batch * in_features * out_features
        torch_gops = (ops_per_forward / torch_mean) / 1e9
        bit_gops = (ops_per_forward / bit_mean) / 1e9

        print(f"Shape: B={batch:4d}, In={in_features:4d}, Out={out_features:4d}")
        print(
            f"  PyTorch F.linear : {torch_ms:8.3f} ms ± {torch_std*1000:.3f} | {torch_gops:7.2f} GOps/s"
        )
        print(
            f"  BitLinear (opt)  : {bit_ms:8.3f} ms ± {bit_std*1000:.3f} | {bit_gops:7.2f} GOps/s"
        )
        if speedup >= 1.0:
            print(f"  Speedup: {speedup:.2f}x faster ✓")
        else:
            print(f"  Speedup: {1/speedup:.2f}x slower ✗")
        print()


if __name__ == "__main__":
    main()

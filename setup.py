#!/usr/bin/env python3
"""
Setup script for BitLab package installation.
"""

import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup, find_packages

def get_extensions():
    """Get C++ extensions for CPU kernels with aggressive optimizations."""
    import platform
    import sys
    import os
    import subprocess
    
    extensions = []
    
    # Base optimization flags
    extra_compile_args = [
        '-O3',                    # Maximum optimization
        '-ffast-math',            # Fast floating point math
        '-funroll-loops',         # Aggressive loop unrolling
        '-fomit-frame-pointer',   # Remove frame pointer for speed
    ]
    
    extra_link_args = []
    include_dirs = []
    library_dirs = []
    
    # Check for OpenMP on macOS
    has_openmp = False
    if platform.system() == 'Darwin':  # macOS
        # Try to find libomp via Homebrew
        try:
            result = subprocess.run(['brew', '--prefix', 'libomp'], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                libomp_prefix = result.stdout.strip()
                if os.path.exists(libomp_prefix):
                    has_openmp = True
                    include_dirs.append(os.path.join(libomp_prefix, 'include'))
                    library_dirs.append(os.path.join(libomp_prefix, 'lib'))
                    extra_compile_args.extend(['-Xpreprocessor', '-fopenmp'])
                    extra_link_args.extend(['-lomp'])
                    print(f"Found libomp at {libomp_prefix}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        if not has_openmp:
            print("Warning: libomp not found. OpenMP disabled. Install with: brew install libomp")
            extra_compile_args.append('-DDISABLE_OPENMP')
        
        # Check if we're on Apple Silicon (ARM64)
        if platform.machine() == 'arm64':
            # ARM64 specific flags - don't use -march=native on ARM
            pass  # Native optimization handled by compiler
        else:
            # x86_64 Intel Mac
            extra_compile_args.extend([
                '-march=native',      # Target native CPU
                '-mtune=native',      # Tune for native CPU
                '-mavx2',             # AVX2 SIMD
                '-mfma',              # FMA instructions
                '-mf16c',             # Half-precision conversion
            ])
        
    elif platform.system() == 'Linux':
        # Linux specific optimizations
        has_openmp = True
        extra_compile_args.extend([
            '-fopenmp',           # OpenMP multithreading
            '-march=native',      # Target native CPU
            '-mtune=native',      # Tune for native CPU
        ])
        extra_link_args.extend(['-lgomp'])
        
        if platform.machine() in ['x86_64', 'AMD64']:
            extra_compile_args.extend([
                '-mavx2',             # AVX2 SIMD
                '-mfma',              # FMA instructions
                '-mf16c',             # Half-precision conversion
            ])
        
    elif platform.system() == 'Windows':
        # Windows/MSVC specific flags
        if 'gcc' in sys.version.lower() or 'clang' in sys.version.lower():
            # MinGW or clang-cl
            has_openmp = True
            extra_compile_args.extend([
                '-fopenmp',
                '-march=native',
                '-mtune=native',
            ])
            extra_link_args.extend(['-lgomp'])
        else:
            # MSVC compiler uses different flags
            has_openmp = True
            extra_compile_args = [
                '/O2',                # Maximum optimization
                '/fp:fast',           # Fast floating point
                '/arch:AVX2',         # AVX2 SIMD
                '/openmp',            # OpenMP
            ]
            extra_link_args = []
    
    extensions.append(
        CppExtension(
            name='bitlab.bnn.functional.kernels.cpu.bitlinear_cpu',
            sources=['src/bitlab/bnn/functional/kernels/cpu/bitlinear_kernel.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
        )
    )
    
    return extensions

if __name__ == "__main__":
    setup(
        name="bitlab",
        version="0.1.0",
        description="A PyTorch library for binary neural networks with efficient quantization",
        author="Your Name",
        author_email="your.email@example.com",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        python_requires=">=3.8",
        install_requires=[
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "numpy>=1.19.0",
        ],
        extras_require={
            "dev": [
                "pytest>=6.0",
                "pytest-cov",
                "black",
                "flake8",
                "mypy",
            ],
            "docs": [
                "sphinx",
                "sphinx-rtd-theme",
            ],
        },
    )
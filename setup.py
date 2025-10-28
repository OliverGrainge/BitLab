#!/usr/bin/env python3
"""
Setup script for BitLab package installation.
"""

import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup, find_packages

def get_extensions():
    """Get C++ extensions for CPU kernels."""
    extensions = []
    
    extensions.append(
        CppExtension(
            name='bitlab.bnn.functional.kernels.cpu.bitlinear_cpu',
            sources=['src/bitlab/bnn/functional/kernels/cpu/bitlinear_kernel.cpp'],
            extra_compile_args=['-O3']
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
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Define C++ extensions
ext_modules = []

# Check if we should build C++ extensions
BUILD_CPP_EXTENSIONS = os.environ.get('BUILD_CPP_EXTENSIONS', '1').lower() in ('1', 'true', 'yes')

if BUILD_CPP_EXTENSIONS:
    ext_modules.extend([
        CppExtension(
            'bitcore.kernels.bindings.bitlinear_int8_pt_pt_cpp',
            [
                'src/bitcore/kernels/bindings/bitlinear_int8_pt_pt.cpp',
            ],
            include_dirs=[],
            extra_compile_args=['-std=c++17'],
        ),
    ])

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
)
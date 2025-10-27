#!/usr/bin/env python3
"""
Build script for C++ extensions.
Usage: python build_extensions.py
"""

import os
import sys
import subprocess
from pathlib import Path

def build_extensions():
    """Build C++ extensions for the project"""
    
    print("Building C++ extensions for BitLab...")
    
    # Set environment variable to enable C++ extensions
    os.environ['BUILD_CPP_EXTENSIONS'] = '1'
    
    try:
        # Run setup.py build_ext
        result = subprocess.run([
            sys.executable, 'setup.py', 'build_ext', '--inplace'
        ], check=True, capture_output=True, text=True)
        
        print("✅ C++ extensions built successfully!")
        print("Output:", result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Failed to build C++ extensions")
        print("Error:", e.stderr)
        return False

def test_import():
    """Test if the C++ extensions can be imported"""
    
    print("\nTesting C++ extension import...")
    
    try:
        from bitcore.kernels.bindings import bitlinear_int8_pt_pt_cpp
        print("✅ C++ extension imported successfully!")
        print(f"Available functions: {dir(bitlinear_int8_pt_pt_cpp)}")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import C++ extension: {e}")
        return False

def main():
    """Main function"""
    
    print("BitLab C++ Extension Builder")
    print("=" * 40)
    
    # Build extensions
    if not build_extensions():
        print("\nBuild failed. Falling back to PyTorch-only mode.")
        return 1
    
    # Test import
    if not test_import():
        print("\nImport test failed. Falling back to PyTorch-only mode.")
        return 1
    
    print("\n🎉 All C++ extensions built and tested successfully!")
    return 0

if __name__ == "__main__":
    exit(main())

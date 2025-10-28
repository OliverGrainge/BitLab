#!/usr/bin/env python3
"""
Setup script for BitLab package installation.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("BitLab Package Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("Error: pyproject.toml not found. Please run this script from the BitLab root directory.")
        sys.exit(1)
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing BitLab package"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("Warning: Development dependencies installation failed, but core package is installed.")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Warning: Tests failed, but package is installed.")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nTo verify installation, run:")
    print("  python -c \"import bitlab; print(f'BitLab version: {bitlab.__version__}')\"")
    print("\nTo run the example:")
    print("  python examples/basic/mlp.py")

if __name__ == "__main__":
    main()

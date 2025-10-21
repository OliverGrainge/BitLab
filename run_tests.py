#!/usr/bin/env python3
"""
Simple test runner script for BitLab
"""
import subprocess
import sys
import os


def run_tests():
    """Run the test suite"""
    print("Running BitLab test suite...")
    print("=" * 50)
    
    # Change to the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode


def run_specific_tests(test_path):
    """Run specific test file or directory"""
    print(f"Running tests in {test_path}...")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ Tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific tests
        test_path = sys.argv[1]
        exit_code = run_specific_tests(test_path)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)

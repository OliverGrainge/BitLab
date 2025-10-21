# BitLab Test Suite

This directory contains the test suite for the BitLab project, focusing on the `bitlayers` module.

## Test Structure

```
tests/
├── __init__.py
├── test_bitlayers/
│   ├── __init__.py
│   ├── test_base.py          # Tests for BitLayerBase
│   ├── test_bitlinear.py     # Tests for BitLinear
│   └── test_registry.py      # Tests for layer registry
├── conftest.py               # Pytest configuration and fixtures
└── README.md                 # This file
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Using pytest directly
pytest

# Using the test runner script
python run_tests.py

# With coverage
pytest --cov=bitlayers --cov-report=html
```

### Run Specific Tests

```bash
# Run tests for a specific module
pytest tests/test_bitlayers/test_bitlinear.py

# Run tests for a specific class
pytest tests/test_bitlayers/test_bitlinear.py::TestBitLinear

# Run tests for a specific method
pytest tests/test_bitlayers/test_bitlinear.py::TestBitLinear::test_init_basic
```

### Run Tests with Different Options

```bash
# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x

# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run with coverage report
pytest --cov=bitlayers --cov-report=term-missing
```

## Test Coverage

The test suite covers:

### BitLayerBase (`test_base.py`)
- ✅ Initialization with default and custom configs
- ✅ Abstract method implementations
- ✅ Mode switching (train/eval)
- ✅ Error handling for unimplemented methods

### BitLinear (`test_bitlinear.py`)
- ✅ Initialization with various parameters
- ✅ Weight initialization methods (Xavier, Kaiming)
- ✅ Quantization parameter setup
- ✅ Forward pass in training and eval modes
- ✅ Gradient flow
- ✅ Parameter registration
- ✅ Different input shapes

### Layer Registry (`test_registry.py`)
- ✅ Layer registration system
- ✅ Decorator functionality
- ✅ Registry persistence
- ✅ Layer instantiation from registry

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- `device`: PyTorch device for tests
- `sample_input_2d/3d`: Sample input tensors
- `default_quant_config`: Default quantization config
- `per_channel_quant_config`: Per-channel quantization config
- `bitlinear_layer`: Pre-configured BitLinear layer
- `reset_random_seed`: Ensures reproducible tests

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `test_*` for functions
2. Use descriptive test names that explain what is being tested
3. Group related tests in classes with `Test*` naming
4. Use fixtures from `conftest.py` when possible
5. Add appropriate assertions and error checking
6. Consider edge cases and error conditions

## Test Categories

Tests are organized by functionality:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Regression Tests**: Test for specific bugs or issues

Use pytest markers to categorize tests:
```python
@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.integration
def test_layer_integration():
    pass
```

## Continuous Integration

The test suite is designed to run in CI environments:

- All tests use CPU-only operations for compatibility
- Random seeds are reset for reproducibility
- Tests are fast and don't require external dependencies
- Coverage reporting is supported

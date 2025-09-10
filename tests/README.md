 # Testing Guide for Legal Document Simplifier

This directory contains comprehensive unit tests for the Legal Document Simplifier project.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_utils.py            # Test utilities and helper functions
├── classification/          # Classification functionality tests
│   ├── __init__.py
│   ├── test_classification_models.py
│   └── test_classification_inference.py
├── summarization/           # Summarization functionality tests
│   ├── __init__.py
│   ├── test_abstractive_summarizer.py
│   └── test_extractive_summarizer.py
├── simplification/          # Simplification functionality tests
│   ├── __init__.py
│   └── test_text_simplifier.py
└── pipeline/                # Pipeline functionality tests
    ├── __init__.py
    ├── test_huggingface_pipeline.py
    └── test_legal_document_pipeline.py
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements.txt
```

### Basic Test Commands

1. **Run all tests:**
   ```bash
   python -m pytest
   # or
   python run_tests.py
   ```

2. **Run specific test modules:**
   ```bash
   python run_tests.py --type classification
   python run_tests.py --type summarization
   python run_tests.py --type simplification
   python run_tests.py --type pipeline
   ```

3. **Run with verbose output:**
   ```bash
   python run_tests.py --verbose
   ```

4. **Run tests in parallel:**
   ```bash
   python run_tests.py --parallel
   ```

5. **Run without coverage:**
   ```bash
   python run_tests.py --no-coverage
   ```

6. **List available tests:**
   ```bash
   python run_tests.py --list
   ```

### Test Categories

Tests are organized by functionality and marked with pytest markers:

- `@pytest.mark.unit` - Unit tests for individual functions/classes
- `@pytest.mark.integration` - Integration tests for component interactions
- `@pytest.mark.slow` - Tests that take longer to run (e.g., model loading)
- `@pytest.mark.gpu` - Tests that require GPU
- `@pytest.mark.classification` - Classification-specific tests
- `@pytest.mark.summarization` - Summarization-specific tests
- `@pytest.mark.simplification` - Simplification-specific tests
- `@pytest.mark.pipeline` - Pipeline-specific tests

### Running Specific Test Categories

```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run only slow tests
python -m pytest -m slow

# Run only GPU tests (if available)
python -m pytest -m gpu
```

## Test Configuration

### pytest.ini

The project uses pytest with the following configuration:
- Test discovery in `tests/` directory
- Coverage reporting with 80% minimum threshold
- Verbose output by default
- HTML coverage report in `htmlcov/` directory

### conftest.py

Contains shared fixtures available to all test modules:
- `temp_config_dir` - Temporary directory with test configuration files
- `mock_tokenizer` - Mock tokenizer for testing
- `mock_model` - Mock model for testing
- `sample_legal_text` - Sample legal text for testing
- `sample_dataset` - Sample dataset for testing
- And many more...

## Writing Tests

### Test Structure

Each test file should follow this structure:

```python
"""
Module docstring describing what this test module covers.
"""
import pytest
from unittest.mock import Mock, patch
from tests.test_utils import create_mock_dataset, MockModel


class TestYourFunctionality:
    """Test class for specific functionality."""
    
    def test_specific_function(self, sample_legal_text, mock_model):
        """Test specific function with given inputs."""
        # Arrange
        expected_result = "expected_output"
        
        # Act
        result = your_function(sample_legal_text, mock_model)
        
        # Assert
        assert result == expected_result
    
    @pytest.mark.parametrize("input_text,expected", [
        ("text1", "result1"),
        ("text2", "result2"),
    ])
    def test_with_multiple_inputs(self, input_text, expected):
        """Test with multiple input/output combinations."""
        result = your_function(input_text)
        assert result == expected
```

### Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use fixtures** for common setup and teardown
4. **Mock external dependencies** (models, APIs, file I/O)
5. **Test edge cases** and error conditions
6. **Use parametrized tests** for multiple input scenarios
7. **Keep tests independent** - each test should be able to run in isolation

### Mocking Guidelines

- Mock expensive operations (model loading, API calls)
- Mock file I/O operations
- Mock random operations for deterministic tests
- Use `@patch` decorator for context-specific mocking
- Use fixtures for reusable mocks

## Coverage

The test suite aims for 80% code coverage across the `src/` directory. Coverage reports are generated in both terminal and HTML formats.

To view HTML coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests are designed to run in CI environments with:
- No GPU requirements (mocked)
- Minimal external dependencies
- Fast execution times
- Deterministic results

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the project root is in Python path
2. **CUDA errors**: Tests mock CUDA availability by default
3. **File not found**: Check that test data files exist
4. **Timeout errors**: Use `@pytest.mark.slow` for long-running tests

### Debug Mode

Run tests with debug output:
```bash
python -m pytest -v -s --tb=long
```

This will show:
- Verbose output (`-v`)
- Print statements (`-s`)
- Full traceback (`--tb=long`)

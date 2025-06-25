# Contributing to White Noise Analysis Toolkit

Thank you for your interest in contributing to the White Noise Analysis Toolkit! This document provides guidelines and information for developers who want to contribute to the project.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation for Development

1. Clone the repository:
```bash
git clone https://github.com/your-username/white-noise-toolkit.git
cd white-noise-toolkit
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install development dependencies:
```bash
pip install pytest pytest-cov black flake8 mypy
```

4. Verify installation:
```bash
python -m white_noise_toolkit.examples.installation_test
```

## Development Workflow

### Code Quality Standards

We maintain high code quality through:

1. **Type Hints**: All public functions should have proper type annotations
2. **Documentation**: All public classes and functions should have comprehensive docstrings
3. **Testing**: All new features should have corresponding tests
4. **Code Style**: We use Black for code formatting

### Running Tests

Run the complete test suite:
```bash
pytest
```

Run specific test modules:
```bash
pytest tests/test_core/
pytest tests/test_utils/
pytest tests/test_synthetic/
pytest tests/test_multi_electrode/
```

Run tests with coverage:
```bash
pytest --cov=white_noise_toolkit --cov-report=html
```

### Code Formatting

Format code with Black:
```bash
black white_noise_toolkit/ tests/
```

Check code style:
```bash
flake8 white_noise_toolkit/
```

Type checking:
```bash
mypy white_noise_toolkit/
```

## Project Structure

```
white_noise_toolkit/
├── core/                    # Core analysis algorithms
│   ├── single_cell.py      # Single cell analysis
│   ├── design_matrix.py    # Design matrix creation
│   ├── filter_extraction.py # Linear filter estimation
│   ├── nonlinearity_estimation.py # Nonlinearity estimation
│   ├── streaming_analyzer.py # Streaming analysis utilities
│   └── exceptions.py       # Custom exceptions
├── utils/                   # Utility modules
│   ├── memory_manager.py   # Memory management
│   ├── io_handlers.py      # File I/O utilities
│   ├── preprocessing.py    # Data preprocessing
│   ├── metrics.py          # Quality metrics
│   └── logging_config.py   # Logging configuration
├── synthetic/               # Synthetic data generation
│   ├── data_generator.py   # Data generation
│   └── validation.py       # Validation utilities
├── multi_electrode/         # Multi-electrode analysis
│   └── __init__.py         # MEA analysis tools
└── examples/               # Examples and tutorials
    └── installation_test.py # Installation test
```

## Adding New Features

### 1. Core Analysis Features

When adding new analysis methods to the core module:

1. Add the implementation to the appropriate module in `core/`
2. Ensure streaming compatibility if applicable
3. Add comprehensive tests in `tests/test_core/`
4. Update the main `SingleCellAnalyzer` class if needed
5. Add documentation and examples

Example:
```python
def new_analysis_method(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    New analysis method description.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data description
    **kwargs
        Additional parameters
        
    Returns
    -------
    dict
        Analysis results
        
    Raises
    ------
    DataValidationError
        If input data is invalid
    """
    # Implementation here
    pass
```

### 2. Utility Functions

When adding utilities:

1. Add to the appropriate module in `utils/`
2. Follow the existing error handling patterns
3. Add tests in `tests/test_utils/`
4. Update `utils/__init__.py` if the utility should be publicly accessible

### 3. Multi-electrode Features

For multi-electrode analysis extensions:

1. Extend the `MultiElectrodeAnalyzer` class
2. Ensure parallel processing compatibility
3. Add tests in `tests/test_multi_electrode/`
4. Consider memory efficiency for large datasets

## Testing Guidelines

### Test Structure

Tests are organized by module:
- `tests/test_core/` - Core analysis functionality
- `tests/test_utils/` - Utility functions
- `tests/test_synthetic/` - Synthetic data generation
- `tests/test_multi_electrode/` - Multi-electrode analysis

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test complete workflows
3. **Property-based Tests**: Test with various input conditions
4. **Performance Tests**: Ensure algorithms meet performance requirements

### Writing Good Tests

```python
class TestNewFeature:
    """Test cases for new feature."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = create_test_data()
    
    def test_basic_functionality(self):
        """Test basic feature functionality."""
        result = new_feature(self.test_data)
        assert result is not None
        assert 'expected_key' in result
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        with pytest.raises(DataValidationError):
            new_feature(invalid_data)
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        result1 = new_feature(self.test_data, seed=42)
        result2 = new_feature(self.test_data, seed=42)
        np.testing.assert_array_equal(result1, result2)
```

## Documentation Guidelines

### Docstring Format

We use NumPy-style docstrings:

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description of the function.
    
    More detailed description if needed. Can span multiple
    paragraphs.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, default=default
        Description of param2
        
    Returns
    -------
    return_type
        Description of return value
        
    Raises
    ------
    ExceptionType
        When this exception is raised
        
    Examples
    --------
    >>> result = function_name(value1, value2)
    >>> print(result)
    expected_output
    
    Notes
    -----
    Additional notes about the function implementation,
    algorithm details, or references.
    """
```

### Code Comments

- Use inline comments sparingly, prefer clear variable names
- Add comments for complex algorithms or non-obvious logic
- Explain the "why" not the "what"

## Performance Considerations

### Memory Management

- Use the `MemoryManager` for large data processing
- Implement streaming where possible
- Consider chunked processing for large datasets
- Monitor memory usage in tests

### Computational Efficiency

- Use NumPy vectorized operations where possible
- Consider Numba JIT compilation for critical loops
- Profile performance-critical code
- Use appropriate data types (float32 vs float64)

## Error Handling

### Exception Hierarchy

Use the custom exceptions defined in `core/exceptions.py`:

- `DataValidationError` - Invalid input data
- `InsufficientDataError` - Not enough data for analysis
- `ProcessingError` - Errors during computation
- `MemoryLimitError` - Memory constraints violated

### Error Messages

Provide clear, actionable error messages:

```python
if len(spike_times) < min_spikes:
    raise InsufficientDataError(
        f"Need at least {min_spikes} spikes for analysis, "
        f"got {len(spike_times)}. Consider using more data "
        f"or reducing filter_length."
    )
```

## Submitting Changes

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite and ensure all tests pass
5. Format code with Black and fix any style issues
6. Update documentation if needed
7. Submit a pull request with a clear description

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Integration tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes generate no new warnings
```

## Release Process

### Version Numbering

We follow semantic versioning (SemVer):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

1. Update version number in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release tag
6. Build and publish to PyPI

## Getting Help

### Communication Channels

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Documentation: Comprehensive guides and API reference

### Code Review

All changes go through code review to ensure:
- Code quality and style consistency
- Test coverage and correctness
- Documentation completeness
- Performance considerations
- Backward compatibility

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you to all contributors who help make this project better! Your contributions are valued and appreciated.

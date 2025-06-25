# White Noise Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A research-grade Python toolkit for white noise analysis of neuronal responses. This toolkit implements streaming algorithms to extract linear-nonlinear (LN) models from stimulus-response data with memory efficiency and numerical stability.

## Overview

The White Noise Analysis Toolkit extracts the receptive field properties of neurons by analyzing their responses to white noise stimuli. It decomposes neural responses into:

1. **Linear Filter**: The neuron's receptive field in space and time
2. **Static Nonlinearity**: The input-output transformation function

The toolkit is designed for computational neuroscience research and supports both single-cell and multi-electrode array (MEA) data analysis.

## Key Features

### 🧠 **Comprehensive Neural Analysis**
- Linear filter extraction (STA and whitened STA methods)
- Static nonlinearity estimation (parametric and non-parametric)
- Complete LN model fitting and validation
- Multi-electrode array support

### ⚡ **High Performance**
- Memory-efficient streaming algorithms
- Configurable memory limits and chunk processing
- Parallel processing for multi-electrode data
- Optimized linear algebra operations

### 🔬 **Research-Grade Quality**
- Robust numerical stability and error handling
- Comprehensive input validation and diagnostics
- Synthetic data generation for method validation
- Statistical testing and ground truth recovery

### 📊 **Flexible Data Support**
- Temporal and spatial-temporal stimuli
- Multiple file formats (HDF5, NumPy, custom)
- Memory-mapped file support for large datasets
- Spike train preprocessing and synchronization

## Installation

### From Source
```bash
git clone https://github.com/your-org/WhiteNoiseToolkit.git
cd WhiteNoiseToolkit
pip install -e .
```

### Dependencies
```bash
pip install numpy scipy scikit-learn matplotlib h5py pandas tqdm pyyaml psutil
```

**Python Requirements**: 3.8+

## Quick Start

### Basic Single-Cell Analysis

```python
import numpy as np
from white_noise_toolkit import SingleCellAnalyzer, SyntheticDataGenerator
from white_noise_toolkit.core.streaming_analyzer import create_stimulus_generator, create_spike_generator

# Generate synthetic data for demonstration
filter_length = 25
t = np.arange(filter_length) * 0.001  # 1ms bins
true_filter = np.exp(-t/0.01) * np.sin(2*np.pi*t/0.005)
true_filter = true_filter / np.linalg.norm(true_filter)

def rectified_linear(x):
    return np.maximum(x + 0.5, 0)

generator = SyntheticDataGenerator(
    filter_true=true_filter,
    nonlinearity_true=rectified_linear,
    noise_level=0.1,
    random_seed=42
)

# Generate stimulus and responses
stimulus = generator.generate_white_noise_stimulus(10000)
spike_counts = generator.generate_responses(stimulus, bin_size=0.001)

# Create analyzer
analyzer = SingleCellAnalyzer(
    bin_size=0.001,
    filter_length=filter_length,
    memory_limit_gb=2.0
)

# Create data generators
stimulus_gen = create_stimulus_generator(stimulus, chunk_size=1000)
spike_gen = create_spike_generator(spike_counts, chunk_size=1000)

# Run analysis
analyzer.fit_streaming(stimulus_gen, spike_gen, progress_bar=True)

# Get results
results = analyzer.get_results()
estimated_filter = results['filter']
nonlinearity_data = results['nonlinearity']

print(f"Filter correlation with ground truth: {np.corrcoef(true_filter, estimated_filter)[0,1]:.3f}")
```

### Working with Real Data

```python
from white_noise_toolkit import load_data, SingleCellAnalyzer

# Load your data
data = load_data('path/to/your/data.h5')
stimulus = data['stimulus']  # Shape: (n_time_bins, ...)
spike_counts = data['spikes']  # Shape: (n_time_bins,)

# Configure analysis
analyzer = SingleCellAnalyzer(
    bin_size=0.008,  # 8ms bins
    filter_length=25,  # 200ms filter
    memory_limit_gb=4.0
)

# Run analysis
from white_noise_toolkit.core.streaming_analyzer import create_stimulus_generator, create_spike_generator

stimulus_gen = create_stimulus_generator(stimulus, chunk_size=2000)
spike_gen = create_spike_generator(spike_counts, chunk_size=2000)

analyzer.fit_streaming(
    stimulus_gen, 
    spike_gen,
    nonlinearity_method='nonparametric',
    extract_both_filters=True
)

# Analyze results
results = analyzer.get_results()
```

## Algorithm Overview

### Linear-Nonlinear (LN) Model

The toolkit implements the standard LN cascade model:

```
Stimulus → Linear Filter → Generator Signal → Static Nonlinearity → Spike Rate
```

#### Mathematical Framework

1. **Design Matrix Construction**: Creates Hankel matrix **X** from stimulus history
2. **Linear Filter Extraction**: 
   - STA: `filter = (X^T @ spikes) / n_spikes`
   - Whitened STA: `filter = (X^T @ X)^(-1) @ (X^T @ spikes)`
3. **Generator Signal**: `g = X @ filter`
4. **Nonlinearity Estimation**: `f(g)` where `f` maps generator signal to firing rate

### Memory-Efficient Streaming

The toolkit processes data in chunks to handle large datasets:

```python
# Streaming accumulation for STA
for chunk in data_chunks:
    design_matrix = create_hankel_matrix(chunk)
    sta_accumulator += design_matrix.T @ spike_chunk
    
# Finalize
sta = sta_accumulator / total_spikes
```

## Advanced Usage

### Multi-Electrode Analysis

```python
from white_noise_toolkit import MultiElectrodeAnalyzer

# Analyze multiple electrodes in parallel
analyzer = MultiElectrodeAnalyzer(
    bin_size=0.008,
    filter_length=25,
    n_jobs=4  # parallel processing
)

results = analyzer.analyze_mea_data(
    stimulus_file='stimulus.h5',
    spike_files=['electrode1.h5', 'electrode2.h5', ...],
    min_spike_count=100
)
```

### Custom Nonlinearities

```python
from white_noise_toolkit import ParametricNonlinearity

# Fit custom parametric model
nonlinearity = ParametricNonlinearity(model_type='sigmoid')
nonlinearity.fit(generator_signal, spike_counts)

# Define custom function
def custom_nonlinearity(x, a, b, c):
    return a * np.exp(b * x) + c

# Use with toolkit...
```

### Validation and Benchmarking

```python
from white_noise_toolkit import GroundTruthRecovery, run_comprehensive_validation

# Test method accuracy
validator = GroundTruthRecovery(random_state=42)
results = validator.test_filter_recovery(
    true_filter,
    stimulus_length=50000,
    noise_level=0.1
)

# Comprehensive validation suite
validation_report = run_comprehensive_validation(
    filter_lengths=[15, 25, 35],
    stimulus_lengths=[10000, 50000, 100000],
    noise_levels=[0.05, 0.1, 0.2]
)
```

## Configuration

### Default Configuration

The toolkit uses YAML configuration files for reproducible analysis:

```yaml
# config/default.yaml
analysis:
  bin_size: 0.008
  filter_length: 25
  chunk_size: 1000
  memory_limit_gb: 8.0

nonlinearity:
  method: 'nonparametric'
  n_bins: 25
  
data_validation:
  min_spike_count: 50
  check_white_noise: true
```

### Custom Configuration

```python
from white_noise_toolkit.utils import load_config

config = load_config('path/to/custom_config.yaml')
analyzer = SingleCellAnalyzer(**config['analysis'])
```

## Data Formats

### Supported Input Formats

- **HDF5** (recommended): `{'stimulus': array, 'spikes': array}`
- **NumPy arrays**: Direct array input
- **Memory-mapped files**: For very large datasets

### Stimulus Data Structure

```python
# Temporal stimulus
stimulus.shape = (n_time_bins,)

# Spatial stimulus (monochrome)
stimulus.shape = (n_time_bins, height, width)

# Spatial stimulus (RGB)
stimulus.shape = (n_time_bins, height, width, 3)
```

### Spike Data Structure

```python
# Single electrode
spikes.shape = (n_time_bins,)

# Multi-electrode
spikes.shape = (n_time_bins, n_electrodes)
```

## Performance Guidelines

### Memory Management

The toolkit automatically manages memory usage:

```python
# Configure memory limits
analyzer = SingleCellAnalyzer(memory_limit_gb=4.0)

# Monitor memory usage
memory_info = analyzer.get_memory_usage()
print(f"Current usage: {memory_info['current_gb']:.1f} GB")
```

### Optimization Tips

1. **Chunk Size**: Balance memory usage vs. processing overhead
   - Small datasets: `chunk_size=1000-2000`
   - Large datasets: `chunk_size=5000-10000`

2. **Filter Length**: Longer filters require more memory
   - Typical range: 15-50 time bins
   - Memory usage scales as `O(filter_length²)`

3. **Spatial Dimensions**: Large images require significant memory
   - Consider downsampling for initial analysis
   - Use `memory_limit_gb` parameter appropriately

## Testing and Validation

### Installation Test

```bash
python -m white_noise_toolkit.examples.installation_test
```

### Run Test Suite

```bash
python -m pytest tests/
```

### Synthetic Data Validation

```python
# Validate method accuracy on known ground truth
from white_noise_toolkit.examples import run_validation_suite
results = run_validation_suite()
```

## API Reference

### Core Classes

- **`SingleCellAnalyzer`**: Main analysis class for individual neurons
- **`MultiElectrodeAnalyzer`**: Parallel analysis for electrode arrays
- **`SyntheticDataGenerator`**: Generate test data with known ground truth
- **`StreamingFilterExtractor`**: Memory-efficient filter extraction
- **`NonparametricNonlinearity`**: Histogram-based nonlinearity estimation
- **`ParametricNonlinearity`**: Model-based nonlinearity fitting

### Utility Functions

- **`create_stimulus_generator()`**: Create data generators from arrays
- **`load_data()`** / **`save_data()`**: File I/O operations
- **`setup_logging()`**: Configure analysis logging
- **`MemoryManager`**: Monitor and control memory usage

## Examples and Tutorials

See the `docs/` directory for comprehensive tutorials:

- **Basic Analysis Tutorial**: Step-by-step single-cell analysis
- **Multi-Electrode Tutorial**: Parallel processing of electrode arrays
- **Advanced Methods**: Custom nonlinearities and validation
- **Performance Optimization**: Memory management and speed tips
- **Real Data Examples**: Working with experimental recordings

## Scientific Background

### Method References

1. **Spike-Triggered Average (STA)**:
   - de Boer, E. & Kuyper, P. (1968). Triggered correlation. IEEE Trans Biomed Eng.

2. **Whitened STA (Maximum Likelihood)**:
   - Paninski, L. (2004). Maximum likelihood estimation of cascade point-process neural encoding models. Network.

3. **LN Model Framework**:
   - Chichilnisky, E.J. (2001). A simple white noise analysis of neuronal light responses. Network.

### Mathematical Foundations

The toolkit implements numerically stable algorithms for:
- **Hankel matrix construction**: Efficient convolution-based design matrices
- **Regularized least squares**: Stable inversion with SVD decomposition
- **Streaming statistics**: Online computation of correlation matrices
- **Nonlinearity estimation**: Adaptive binning and interpolation

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Development Setup

```bash
git clone https://github.com/your-org/WhiteNoiseToolkit.git
cd WhiteNoiseToolkit
pip install -e ".[dev]"
pre-commit install
```

## License

This project is licensed under the MIT License - see `LICENSE` file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{white_noise_toolkit_2025,
  title={White Noise Analysis Toolkit: A Python Package for Neural Receptive Field Analysis},
  author={White Noise Analysis Team},
  year={2025},
  url={https://github.com/your-org/WhiteNoiseToolkit}
}
```

## Support

- **Documentation**: [https://whitenoise-toolkit.readthedocs.io](https://whitenoise-toolkit.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/WhiteNoiseToolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/WhiteNoiseToolkit/discussions)

---

**Note**: This toolkit is designed for research purposes. For production neuroscience applications, additional validation and testing may be required.

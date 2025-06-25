# Tutorial 1: Basic White Noise Analysis

This tutorial provides a comprehensive introduction to white noise analysis using the White Noise Analysis Toolkit. You'll learn the fundamental concepts and perform your first analysis.

## Table of Contents

1. [Introduction to White Noise Analysis](#introduction)
2. [The Linear-Nonlinear (LN) Model](#ln-model)
3. [Setting Up Your Environment](#setup)
4. [Your First Analysis](#first-analysis)
5. [Understanding the Results](#understanding-results)
6. [Working with Real Data](#real-data)
7. [Best Practices](#best-practices)

## Introduction to White Noise Analysis {#introduction}

White noise analysis is a fundamental technique in computational neuroscience for characterizing neural receptive fields. It reveals how neurons respond to visual, auditory, or other sensory stimuli by analyzing their responses to random (white noise) inputs.

### Why White Noise?

1. **Unbiased Sampling**: White noise uniformly samples the stimulus space
2. **Linear System Identification**: Optimal for extracting linear filters
3. **Statistical Power**: Efficient data collection for receptive field mapping
4. **Nonlinearity Detection**: Reveals static nonlinear transformations

### What You'll Learn

- Extract linear receptive fields (filters) from neural responses
- Estimate static nonlinearities
- Validate results with synthetic data
- Handle memory-efficient computation for large datasets

## The Linear-Nonlinear (LN) Model {#ln-model}

The LN model decomposes neural responses into two sequential operations:

```
Stimulus → [Linear Filter] → Generator Signal → [Static Nonlinearity] → Firing Rate
```

### Mathematical Framework

1. **Linear Stage**: `g(t) = ∫ k(τ) s(t-τ) dτ`
   - `s(t)`: stimulus at time t
   - `k(τ)`: linear filter (receptive field)
   - `g(t)`: generator signal

2. **Nonlinear Stage**: `r(t) = f(g(t))`
   - `f()`: static nonlinearity
   - `r(t)`: instantaneous firing rate

### Key Concepts

- **Receptive Field**: The linear filter `k(τ)` represents the neuron's sensitivity to stimulus history
- **Generator Signal**: The projection of stimulus onto the receptive field
- **Static Nonlinearity**: The input-output transformation from generator signal to firing rate

## Setting Up Your Environment {#setup}

### Installation

First, ensure you have the toolkit installed:

```bash
pip install -e /path/to/WhiteNoiseToolkit
```

### Import Essential Modules

```python
import numpy as np
import matplotlib.pyplot as plt
from white_noise_toolkit import (
    SingleCellAnalyzer,
    SyntheticDataGenerator,
    setup_logging
)
from white_noise_toolkit.core.streaming_analyzer import (
    create_stimulus_generator,
    create_spike_generator
)

# Set up logging to see what's happening
setup_logging(level='INFO')
```

### Configure Matplotlib for Better Plots

```python
plt.style.use('seaborn-v0_8')  # Clean plotting style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

## Your First Analysis {#first-analysis}

Let's start with synthetic data where we know the ground truth, making it easy to validate our results.

### Step 1: Create Synthetic Data

```python
import numpy as np

# Define analysis parameters
bin_size = 0.001  # 1ms time bins
filter_length = 25  # 25ms filter
sampling_rate = 1000  # Hz

# Create a realistic receptive field
t = np.arange(filter_length) * bin_size  # Time vector in seconds
true_filter = np.exp(-t/0.01) * np.sin(2*np.pi*t/0.005)
true_filter = true_filter / np.linalg.norm(true_filter)  # Normalize

# Define the true nonlinearity (rectified linear with threshold)
def true_nonlinearity(x):
    return np.maximum(x + 0.3, 0)  # Threshold at -0.3

# Create synthetic data generator
generator = SyntheticDataGenerator(
    filter_true=true_filter,
    nonlinearity_true=true_nonlinearity,
    noise_level=0.1,  # 10% noise
    random_seed=42   # For reproducibility
)

print("Created synthetic data generator with known ground truth")
```

### Step 2: Generate Stimulus and Responses

```python
# Generate white noise stimulus
n_samples = 50000  # 50 seconds of data at 1kHz
stimulus = generator.generate_white_noise_stimulus(n_samples)

# Generate neural responses
spike_counts = generator.generate_responses(stimulus, bin_size=bin_size)

# Basic statistics
total_spikes = np.sum(spike_counts)
mean_rate = total_spikes / (n_samples * bin_size)

print(f"Generated {n_samples} samples ({n_samples * bin_size:.1f} seconds)")
print(f"Total spikes: {total_spikes}")
print(f"Mean firing rate: {mean_rate:.1f} Hz")
print(f"Stimulus statistics: mean={np.mean(stimulus):.3f}, std={np.std(stimulus):.3f}")
```

### Step 3: Set Up the Analyzer

```python
# Create the analyzer
analyzer = SingleCellAnalyzer(
    bin_size=bin_size,
    filter_length=filter_length,
    memory_limit_gb=2.0  # Adjust based on your system
)

# Validate configuration
validation = analyzer.validate_configuration()
if not validation['valid']:
    print("Configuration issues:")
    for error in validation['errors']:
        print(f"  ERROR: {error}")
for warning in validation['warnings']:
    print(f"  WARNING: {warning}")
```

### Step 4: Create Data Generators

The toolkit uses generators for memory-efficient processing:

```python
# Create generators for streaming analysis
chunk_size = 2000  # Process 2000 samples at a time

stimulus_gen = create_stimulus_generator(stimulus, chunk_size=chunk_size)
spike_gen = create_spike_generator(spike_counts, chunk_size=chunk_size)

print(f"Created generators with chunk size: {chunk_size}")
```

### Step 5: Run the Analysis

```python
# Perform the complete analysis
analyzer.fit_streaming(
    stimulus_gen, 
    spike_gen,
    chunk_size=chunk_size,
    nonlinearity_method='nonparametric',  # Histogram-based estimation
    extract_both_filters=True,  # Compute both STA and whitened STA
    progress_bar=True,  # Show progress
    n_bins=25  # Number of bins for nonlinearity estimation
)

print("Analysis completed successfully!")
```

### Step 6: Extract Results

```python
# Get the complete results
results = analyzer.get_results()

# Extract components
estimated_filter = results['filter']  # Best filter estimate (whitened STA)
sta = results['sta']  # Spike-triggered average
nonlinearity_data = results['nonlinearity']  # Nonlinearity estimate
metadata = results['metadata']  # Analysis parameters and diagnostics

print("Results extracted:")
print(f"  Filter shape: {estimated_filter.shape}")
print(f"  STA shape: {sta.shape}")
print(f"  Nonlinearity bins: {len(nonlinearity_data['x_values'])}")
```

## Understanding the Results {#understanding-results}

### Visualizing the Linear Filter

```python
# Create time axis for plotting
t_ms = np.arange(filter_length) * bin_size * 1000  # Convert to milliseconds

# Plot filter comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Filter comparison
axes[0].plot(t_ms, true_filter, 'k-', linewidth=2, label='True Filter')
axes[0].plot(t_ms, sta, 'b--', linewidth=2, label='STA')
axes[0].plot(t_ms, estimated_filter, 'r-', linewidth=2, label='Whitened STA')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Filter Weight')
axes[0].set_title('Receptive Field Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compute correlations
sta_corr = np.corrcoef(true_filter, sta)[0, 1]
filter_corr = np.corrcoef(true_filter, estimated_filter)[0, 1]

axes[0].text(0.02, 0.98, 
            f'STA correlation: {sta_corr:.3f}\nWhitened STA correlation: {filter_corr:.3f}',
            transform=axes[0].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Filter power spectrum
freqs = np.fft.fftfreq(filter_length, bin_size)[:filter_length//2]
true_power = np.abs(np.fft.fft(true_filter))[:filter_length//2]
est_power = np.abs(np.fft.fft(estimated_filter))[:filter_length//2]

axes[1].semilogy(freqs, true_power, 'k-', linewidth=2, label='True Filter')
axes[1].semilogy(freqs, est_power, 'r-', linewidth=2, label='Estimated Filter')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power')
axes[1].set_title('Filter Frequency Response')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Visualizing the Nonlinearity

```python
# Extract nonlinearity data
x_vals = nonlinearity_data['x_values']
y_vals = nonlinearity_data['y_values']

# Create fine-grained true nonlinearity for comparison
x_fine = np.linspace(np.min(x_vals), np.max(x_vals), 200)
y_true = true_nonlinearity(x_fine)

# Plot nonlinearity comparison
plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_true, 'k-', linewidth=3, label='True Nonlinearity')
plt.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=6, label='Estimated Nonlinearity')
plt.xlabel('Generator Signal')
plt.ylabel('Firing Rate (Hz)')
plt.title('Static Nonlinearity Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics
mean_error = np.mean(np.abs(y_vals - true_nonlinearity(x_vals)))
plt.text(0.02, 0.98, f'Mean absolute error: {mean_error:.2f} Hz',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

### Analyzing Performance Metrics

```python
# Extract performance metrics
if 'performance_metrics' in results:
    perf = results['performance_metrics']
    
    print("Performance Metrics:")
    print(f"  Total spikes processed: {perf.get('total_spikes', 'N/A')}")
    print(f"  Mean firing rate: {perf.get('spike_rate_hz', 'N/A'):.1f} Hz")
    print(f"  Filter SNR estimate: {perf.get('filter_snr_estimate', 'N/A'):.2f}")
    print(f"  Memory usage: {perf.get('memory_usage', {}).get('peak_gb', 'N/A'):.2f} GB")

# Analysis metadata
print("\nAnalysis Configuration:")
for key, value in metadata.items():
    if key not in ['filter_comparison', 'filter_validation']:
        print(f"  {key}: {value}")
```

## Working with Real Data {#real-data}

### Loading Experimental Data

```python
from white_noise_toolkit import load_data

# Load your experimental data
# Expected format: HDF5 file with 'stimulus' and 'spikes' datasets
try:
    data = load_data('path/to/your/experiment.h5')
    stimulus = data['stimulus']
    spike_counts = data['spikes']
    
    print(f"Loaded data:")
    print(f"  Stimulus shape: {stimulus.shape}")
    print(f"  Spikes shape: {spike_counts.shape}")
    print(f"  Total spikes: {np.sum(spike_counts)}")
    
except FileNotFoundError:
    print("Real data file not found. Continuing with synthetic data example...")
    # Continue with synthetic data for this tutorial
```

### Data Quality Checks

```python
from white_noise_toolkit.utils.preprocessing import StimulusProcessor

# Create stimulus processor for validation
processor = StimulusProcessor()

# Check stimulus statistics
stimulus_stats = processor.compute_statistics(stimulus)
print("Stimulus Statistics:")
print(f"  Mean: {stimulus_stats['mean']:.4f} (should be ≈ 0)")
print(f"  Std: {stimulus_stats['std']:.4f} (should be ≈ 1)")
print(f"  Skewness: {stimulus_stats['skewness']:.4f}")
print(f"  Kurtosis: {stimulus_stats['kurtosis']:.4f}")

# Check for potential issues
if abs(stimulus_stats['mean']) > 0.1:
    print("⚠️  WARNING: Stimulus mean is not close to zero")
if abs(stimulus_stats['std'] - 1.0) > 0.2:
    print("⚠️  WARNING: Stimulus std is not close to 1")

# Check spike statistics
spike_rate = np.sum(spike_counts) / (len(spike_counts) * bin_size)
if spike_rate < 1.0:
    print("⚠️  WARNING: Very low firing rate - results may be unreliable")
elif spike_rate > 1000.0:
    print("⚠️  WARNING: Very high firing rate - check time units")
```

## Best Practices {#best-practices}

### 1. Memory Management

```python
# Monitor memory usage
memory_info = analyzer.get_memory_usage()
print(f"Current memory usage: {memory_info['current_gb']:.1f} GB")

# Adjust chunk size based on available memory
available_gb = memory_info['available_gb']
if available_gb < 2.0:
    chunk_size = 1000  # Smaller chunks for limited memory
elif available_gb > 8.0:
    chunk_size = 5000  # Larger chunks for better performance
```

### 2. Parameter Selection

```python
# Filter length selection
# Rule of thumb: capture significant temporal dependencies
# Too short: miss temporal structure
# Too long: overfitting and memory issues

# Check autocorrelation to guide filter length
def plot_sta_evolution(stimulus, spikes, max_length=50):
    """Plot how STA evolves with filter length"""
    correlations = []
    lengths = range(5, max_length+1, 5)
    
    for length in lengths:
        # Quick STA calculation
        design_matrix = create_design_matrix_batch(stimulus[:5000], length)
        sta = (design_matrix.T @ spikes[:5000]) / np.sum(spikes[:5000])
        
        # Correlation with final STA (using max length)
        if length == max_length:
            final_sta = sta
            correlations.append(1.0)
        else:
            # Pad shorter STA for comparison
            padded_sta = np.pad(sta, (0, max_length - length), 'constant')
            corr = np.corrcoef(padded_sta, final_sta)[0, 1]
            correlations.append(corr)
    
    plt.figure(figsize=(8, 5))
    plt.plot(lengths, correlations, 'bo-')
    plt.xlabel('Filter Length (time bins)')
    plt.ylabel('Correlation with Full STA')
    plt.title('STA Convergence vs Filter Length')
    plt.grid(True, alpha=0.3)
    plt.show()

# Uncomment to run this analysis:
# plot_sta_evolution(stimulus, spike_counts)
```

### 3. Validation and Diagnostics

```python
# Cross-validation: split data and compare results
def cross_validate_analysis(stimulus, spikes, n_splits=5):
    """Perform cross-validation of the analysis"""
    n_samples = len(stimulus)
    split_size = n_samples // n_splits
    
    correlations = []
    
    for i in range(n_splits):
        # Create train/test split
        test_start = i * split_size
        test_end = (i + 1) * split_size
        
        train_stimulus = np.concatenate([
            stimulus[:test_start], 
            stimulus[test_end:]
        ])
        train_spikes = np.concatenate([
            spikes[:test_start], 
            spikes[test_end:]
        ])
        
        # Fit on training data
        analyzer_cv = SingleCellAnalyzer(
            bin_size=bin_size,
            filter_length=filter_length,
            memory_limit_gb=1.0
        )
        
        train_stim_gen = create_stimulus_generator(train_stimulus, chunk_size=1000)
        train_spike_gen = create_spike_generator(train_spikes, chunk_size=1000)
        
        analyzer_cv.fit_streaming(
            train_stim_gen, train_spike_gen, 
            progress_bar=False
        )
        
        cv_results = analyzer_cv.get_results()
        cv_filter = cv_results['filter']
        
        # Compare with main result
        corr = np.corrcoef(estimated_filter, cv_filter)[0, 1]
        correlations.append(corr)
    
    return np.array(correlations)

# Run cross-validation (uncomment to use)
# cv_corrs = cross_validate_analysis(stimulus, spike_counts)
# print(f"Cross-validation filter correlations: {cv_corrs.mean():.3f} ± {cv_corrs.std():.3f}")
```

### 4. Troubleshooting Common Issues

```python
# Check for common problems

def diagnose_analysis(analyzer, results):
    """Diagnose potential issues with analysis results"""
    issues = []
    warnings = []
    
    # Check filter properties
    filter_est = results['filter']
    filter_norm = np.linalg.norm(filter_est)
    filter_max = np.max(np.abs(filter_est))
    
    if filter_norm < 1e-6:
        issues.append("Filter norm is very small - possible numerical issues")
    
    if filter_max == filter_est[0] or filter_max == filter_est[-1]:
        warnings.append("Filter peak at boundary - consider longer filter")
    
    # Check nonlinearity
    if 'nonlinearity' in results:
        nl_data = results['nonlinearity']
        y_range = np.max(nl_data['y_values']) - np.min(nl_data['y_values'])
        if y_range < 1.0:
            warnings.append("Nonlinearity has small dynamic range")
    
    # Check spike count
    metadata = results.get('metadata', {})
    if 'performance_metrics' in results:
        total_spikes = results['performance_metrics'].get('total_spikes', 0)
        if total_spikes < 100:
            issues.append(f"Very few spikes ({total_spikes}) - results unreliable")
        elif total_spikes < 1000:
            warnings.append(f"Low spike count ({total_spikes}) - consider more data")
    
    return issues, warnings

# Run diagnostics
issues, warnings = diagnose_analysis(analyzer, results)

if issues:
    print("❌ ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")

if warnings:
    print("⚠️  WARNINGS:")
    for warning in warnings:
        print(f"  {warning}")

if not issues and not warnings:
    print("✅ Analysis looks good!")
```

## Summary

In this tutorial, you learned:

1. **Theoretical Background**: The LN model and white noise analysis principles
2. **Practical Implementation**: How to use the toolkit for complete analysis
3. **Result Interpretation**: Understanding filters, nonlinearities, and metrics
4. **Best Practices**: Memory management, parameter selection, and validation
5. **Troubleshooting**: Common issues and diagnostic techniques

### Next Steps

- **Tutorial 2**: Advanced analysis techniques and custom nonlinearities
- **Tutorial 3**: Multi-electrode array analysis
- **Tutorial 4**: Working with spatial-temporal stimuli
- **Tutorial 5**: Performance optimization and large-scale analysis

### Additional Resources

- **API Reference**: Complete documentation of all classes and functions
- **Example Scripts**: Ready-to-run analysis examples
- **Validation Suite**: Test the toolkit's accuracy on your system

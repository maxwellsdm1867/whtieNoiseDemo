# Tutorial 2: Advanced Analysis Techniques

This tutorial covers advanced features of the White Noise Analysis Toolkit, including custom nonlinearities, spatial-temporal analysis, and performance optimization.

## Table of Contents

1. [Parametric Nonlinearity Models](#parametric-models)
2. [Spatial-Temporal Receptive Fields](#spatial-temporal)
3. [Custom Nonlinearity Functions](#custom-nonlinearities)
4. [Advanced Filter Extraction](#advanced-filters)
5. [Performance Optimization](#performance)
6. [Statistical Validation](#validation)
7. [Real-World Examples](#examples)

## Parametric Nonlinearity Models {#parametric-models}

While nonparametric (histogram-based) estimation is robust, parametric models offer interpretable parameters and better extrapolation.

### Available Parametric Models

The toolkit supports several standard nonlinearity models:

```python
import numpy as np
import matplotlib.pyplot as plt
from white_noise_toolkit import ParametricNonlinearity

# 1. Exponential Model: f(x) = a * exp(b * x) + c
exponential = ParametricNonlinearity(model_type='exponential')

# 2. Cumulative Normal: f(x) = a * Φ((x - μ)/σ) + c
cumulative_normal = ParametricNonlinearity(model_type='cumulative_normal')

# 3. Sigmoid: f(x) = a / (1 + exp(-b * (x - c))) + d
sigmoid = ParametricNonlinearity(model_type='sigmoid')

print("Available parametric models:")
print("  exponential: a * exp(b * x) + c")
print("  cumulative_normal: a * Φ((x - μ)/σ) + c")
print("  sigmoid: a / (1 + exp(-b * (x - c))) + d")
```

### Fitting Parametric Models

```python
from white_noise_toolkit import SingleCellAnalyzer, SyntheticDataGenerator

# Create synthetic data with known parametric nonlinearity
def sigmoid_nonlinearity(x):
    return 50 / (1 + np.exp(-3 * (x - 0.1))) + 2

# Generate data
filter_length = 25
t = np.arange(filter_length) * 0.001
true_filter = np.exp(-t/0.008) * np.sin(2*np.pi*t/0.004)
true_filter = true_filter / np.linalg.norm(true_filter)

generator = SyntheticDataGenerator(
    filter_true=true_filter,
    nonlinearity_true=sigmoid_nonlinearity,
    noise_level=0.05,
    random_seed=42
)

stimulus = generator.generate_white_noise_stimulus(30000)
spike_counts = generator.generate_responses(stimulus, bin_size=0.001)

# Analyze with parametric fitting
analyzer = SingleCellAnalyzer(
    bin_size=0.001,
    filter_length=filter_length,
    memory_limit_gb=2.0
)

from white_noise_toolkit.core.streaming_analyzer import create_stimulus_generator, create_spike_generator

stimulus_gen = create_stimulus_generator(stimulus, chunk_size=2000)
spike_gen = create_spike_generator(spike_counts, chunk_size=2000)

# Fit with parametric nonlinearity
analyzer.fit_streaming(
    stimulus_gen, spike_gen,
    nonlinearity_method='parametric',
    model_type='sigmoid',  # Specify the model type
    progress_bar=True
)

results = analyzer.get_results()
```

### Comparing Parametric vs Nonparametric

```python
# Fit both models for comparison
analyzer_nonparam = SingleCellAnalyzer(bin_size=0.001, filter_length=filter_length)
analyzer_param = SingleCellAnalyzer(bin_size=0.001, filter_length=filter_length)

# Create fresh generators
stim_gen1 = create_stimulus_generator(stimulus, chunk_size=2000)
spike_gen1 = create_spike_generator(spike_counts, chunk_size=2000)
stim_gen2 = create_stimulus_generator(stimulus, chunk_size=2000)
spike_gen2 = create_spike_generator(spike_counts, chunk_size=2000)

# Fit nonparametric
analyzer_nonparam.fit_streaming(
    stim_gen1, spike_gen1,
    nonlinearity_method='nonparametric',
    n_bins=30
)

# Fit parametric
analyzer_param.fit_streaming(
    stim_gen2, spike_gen2,
    nonlinearity_method='parametric',
    model_type='sigmoid'
)

# Extract results
results_nonparam = analyzer_nonparam.get_results()
results_param = analyzer_param.get_results()

# Plot comparison
x_range = np.linspace(-1, 1, 200)
y_true = sigmoid_nonlinearity(x_range)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Nonparametric
nl_np = results_nonparam['nonlinearity']
plt.plot(nl_np['x_values'], nl_np['y_values'], 'bo-', label='Nonparametric', markersize=4)
plt.plot(x_range, y_true, 'k-', linewidth=2, label='True Function')
plt.xlabel('Generator Signal')
plt.ylabel('Firing Rate (Hz)')
plt.title('Nonparametric Estimation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Parametric
nl_p = results_param['nonlinearity']
plt.plot(nl_p['x_values'], nl_p['y_values'], 'r-', linewidth=2, label='Parametric Fit')
plt.plot(x_range, y_true, 'k-', linewidth=2, label='True Function')
plt.xlabel('Generator Signal')
plt.ylabel('Firing Rate (Hz)')
plt.title('Parametric Estimation (Sigmoid)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add parameter information
params = nl_p['parameters']
plt.text(0.02, 0.98, f'Parameters: {params}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

## Spatial-Temporal Receptive Fields {#spatial-temporal}

Many neurons have receptive fields that vary in both space and time. The toolkit handles spatial-temporal analysis seamlessly.

### Generating Spatial-Temporal Data

```python
# Create a spatial-temporal filter
def create_spatiotemporal_filter(height=10, width=10, filter_length=15):
    """Create a separable spatial-temporal filter"""
    
    # Temporal component (difference of Gaussians)
    t = np.arange(filter_length) * 0.008  # 8ms bins
    temporal = (np.exp(-t/0.05) - 0.5 * np.exp(-t/0.1)) * np.sin(2*np.pi*t/0.04)
    temporal = temporal / np.linalg.norm(temporal)
    
    # Spatial component (Gabor filter)
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    X, Y = np.meshgrid(x, y)
    
    # Gabor parameters
    freq = 0.5
    theta = np.pi/4  # 45 degrees
    sigma = 1.0
    
    # Rotate coordinates
    x_rot = X * np.cos(theta) + Y * np.sin(theta)
    y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Gabor function
    spatial = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2)) * np.cos(2 * np.pi * freq * x_rot)
    spatial = spatial / np.linalg.norm(spatial)
    
    # Combine temporal and spatial (separable filter)
    spatiotemporal = np.outer(temporal, spatial.ravel())
    return spatiotemporal.ravel(), temporal, spatial

# Create the filter
height, width = 8, 8
filter_length = 20
st_filter, temporal_component, spatial_component = create_spatiotemporal_filter(
    height, width, filter_length
)

print(f"Created spatial-temporal filter:")
print(f"  Spatial dimensions: {height} x {width}")
print(f"  Temporal length: {filter_length}")
print(f"  Total filter size: {len(st_filter)}")
```

### Generating Spatial-Temporal Stimulus

```python
# Generate spatial white noise stimulus
n_frames = 25000  # Number of time frames
spatial_dims = (height, width)

# Create spatial-temporal data generator
generator_st = SyntheticDataGenerator(
    filter_true=st_filter,
    nonlinearity_true=lambda x: np.maximum(x + 0.2, 0),
    noise_level=0.1,
    random_seed=42
)

# Generate spatial white noise stimulus
stimulus_st = generator_st.generate_white_noise_stimulus(
    n_time_bins=n_frames,
    spatial_dims=spatial_dims,
    n_colors=1,
    contrast_std=1.0
)

print(f"Generated spatial-temporal stimulus: {stimulus_st.shape}")

# Generate responses
spike_counts_st = generator_st.generate_responses(stimulus_st, bin_size=0.008)
print(f"Generated spikes: {np.sum(spike_counts_st)} total")
```

### Analyzing Spatial-Temporal Data

```python
# Create analyzer for spatial-temporal data
analyzer_st = SingleCellAnalyzer(
    bin_size=0.008,
    filter_length=filter_length,
    spatial_dims=spatial_dims,
    n_colors=1,
    memory_limit_gb=3.0  # May need more memory for spatial data
)

# Create generators
stim_gen_st = create_stimulus_generator(stimulus_st, chunk_size=1500)
spike_gen_st = create_spike_generator(spike_counts_st, chunk_size=1500)

# Run analysis
analyzer_st.fit_streaming(
    stim_gen_st, spike_gen_st,
    nonlinearity_method='nonparametric',
    extract_both_filters=True,
    progress_bar=True
)

results_st = analyzer_st.get_results()
estimated_st_filter = results_st['filter']

print(f"Estimated spatial-temporal filter shape: {estimated_st_filter.shape}")
```

### Visualizing Spatial-Temporal Results

```python
# Reshape filters for visualization
def reshape_st_filter(filter_1d, filter_length, height, width):
    """Reshape 1D filter to (time, height, width)"""
    return filter_1d.reshape(filter_length, height, width)

true_st_reshaped = reshape_st_filter(st_filter, filter_length, height, width)
est_st_reshaped = reshape_st_filter(estimated_st_filter, filter_length, height, width)

# Plot spatial-temporal filter evolution
fig, axes = plt.subplots(2, min(6, filter_length), figsize=(15, 6))

time_indices = np.linspace(0, filter_length-1, min(6, filter_length)).astype(int)

for i, t_idx in enumerate(time_indices):
    # True filter
    im1 = axes[0, i].imshow(true_st_reshaped[t_idx], cmap='RdBu_r', 
                           vmin=-np.max(np.abs(true_st_reshaped)),
                           vmax=np.max(np.abs(true_st_reshaped)))
    axes[0, i].set_title(f't = {t_idx * 8:.0f} ms')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    
    # Estimated filter
    im2 = axes[1, i].imshow(est_st_reshaped[t_idx], cmap='RdBu_r',
                           vmin=-np.max(np.abs(est_st_reshaped)),
                           vmax=np.max(np.abs(est_st_reshaped)))
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel('True Filter')
axes[1, 0].set_ylabel('Estimated')

# Add colorbars
plt.colorbar(im1, ax=axes[0, -1])
plt.colorbar(im2, ax=axes[1, -1])

plt.suptitle('Spatial-Temporal Receptive Field Evolution')
plt.tight_layout()
plt.show()

# Compare temporal and spatial components
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Extract temporal profiles by averaging over space
true_temporal = np.mean(true_st_reshaped.reshape(filter_length, -1), axis=1)
est_temporal = np.mean(est_st_reshaped.reshape(filter_length, -1), axis=1)

t_ms = np.arange(filter_length) * 8
plt.plot(t_ms, true_temporal, 'k-', linewidth=2, label='True')
plt.plot(t_ms, est_temporal, 'r-', linewidth=2, label='Estimated')
plt.xlabel('Time (ms)')
plt.ylabel('Mean Response')
plt.title('Temporal Profile')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Extract spatial profiles by averaging over time
true_spatial = np.mean(true_st_reshaped, axis=0)
est_spatial = np.mean(est_st_reshaped, axis=0)

plt.imshow(true_spatial, cmap='RdBu_r', alpha=0.7)
plt.contour(est_spatial, colors='white', linewidths=2)
plt.title('Spatial Profile\n(True: color, Estimated: contours)')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
```

## Custom Nonlinearity Functions {#custom-nonlinearities}

For specialized applications, you can define custom nonlinearity models:

### Creating Custom Parametric Models

```python
from scipy.optimize import minimize
from white_noise_toolkit.core.nonlinearity_estimation import ParametricNonlinearity

class CustomNonlinearity:
    """Custom nonlinearity with user-defined function"""
    
    def __init__(self, function, initial_params, param_bounds=None):
        """
        Parameters
        ----------
        function : callable
            Function of form f(x, *params)
        initial_params : array_like
            Initial parameter guess
        param_bounds : list of tuples, optional
            Parameter bounds for optimization
        """
        self.function = function
        self.initial_params = np.array(initial_params)
        self.param_bounds = param_bounds
        self.fitted_params = None
        self.fitted = False
    
    def fit(self, generator_signal, spike_counts):
        """Fit the custom nonlinearity to data"""
        
        def objective(params):
            """Negative log-likelihood for Poisson model"""
            predicted = self.function(generator_signal, *params)
            predicted = np.maximum(predicted, 1e-10)  # Avoid log(0)
            
            # Poisson negative log-likelihood
            return -np.sum(spike_counts * np.log(predicted) - predicted)
        
        # Optimize parameters
        result = minimize(
            objective, 
            self.initial_params,
            bounds=self.param_bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            self.fitted_params = result.x
            self.fitted = True
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")
    
    def predict(self, generator_signal):
        """Predict firing rates for given generator signals"""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.function(generator_signal, *self.fitted_params)

# Example: Double exponential nonlinearity
def double_exponential(x, a1, b1, a2, b2, c):
    """Double exponential: a1*exp(b1*x) + a2*exp(b2*x) + c"""
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

# Create and fit custom nonlinearity
custom_nl = CustomNonlinearity(
    function=double_exponential,
    initial_params=[10, 2, 5, -1, 1],  # [a1, b1, a2, b2, c]
    param_bounds=[(0, 100), (0, 10), (0, 100), (-10, 0), (0, 50)]
)

# Example usage (with synthetic data from previous section)
# Generate test data with double exponential nonlinearity
def true_double_exp(x):
    return double_exponential(x, 15, 3, 8, -2, 2)

# Create synthetic data
generator_custom = SyntheticDataGenerator(
    filter_true=true_filter,
    nonlinearity_true=true_double_exp,
    noise_level=0.08,
    random_seed=42
)

stimulus_custom = generator_custom.generate_white_noise_stimulus(20000)
spikes_custom = generator_custom.generate_responses(stimulus_custom, bin_size=0.001)

# Fit the custom model manually
# First extract filter using standard analysis
analyzer_custom = SingleCellAnalyzer(bin_size=0.001, filter_length=filter_length)
stim_gen = create_stimulus_generator(stimulus_custom, chunk_size=2000)
spike_gen = create_spike_generator(spikes_custom, chunk_size=2000)

analyzer_custom.fit_streaming(stim_gen, spike_gen, nonlinearity_method='nonparametric')
results_custom = analyzer_custom.get_results()

# Extract generator signal manually for custom fitting
from white_noise_toolkit.core.design_matrix import create_design_matrix_batch

design_matrix = create_design_matrix_batch(stimulus_custom, filter_length=filter_length)
generator_signal = design_matrix @ results_custom['filter']

# Fit custom nonlinearity
custom_nl.fit(generator_signal, spikes_custom)

print(f"Fitted parameters: {custom_nl.fitted_params}")
print(f"True parameters: [15, 3, 8, -2, 2]")
```

## Advanced Filter Extraction {#advanced-filters}

### Regularization Techniques

```python
from white_noise_toolkit.core.filter_extraction import StreamingFilterExtractor

# Custom filter extraction with different regularization
def extract_filter_with_regularization(stimulus, spikes, filter_length, reg_strength=1e-6):
    """Extract filter with custom regularization"""
    
    extractor = StreamingFilterExtractor()
    
    # Create design matrix
    design_matrix = create_design_matrix_batch(stimulus, filter_length=filter_length)
    
    # Accumulate statistics
    extractor.compute_sta_streaming(design_matrix, spikes)
    extractor.compute_whitened_sta_streaming(design_matrix, spikes)
    
    # Extract with custom regularization
    filter_reg = extractor.finalize_whitened_sta(regularization=reg_strength)
    
    return filter_reg

# Compare different regularization strengths
reg_strengths = [1e-8, 1e-6, 1e-4, 1e-2]
filters_reg = []

for reg in reg_strengths:
    filter_reg = extract_filter_with_regularization(
        stimulus, spike_counts, filter_length, reg_strength=reg
    )
    filters_reg.append(filter_reg)

# Plot comparison
plt.figure(figsize=(12, 8))
t_ms = np.arange(filter_length) * 1  # 1ms bins

for i, (reg, filt) in enumerate(zip(reg_strengths, filters_reg)):
    plt.subplot(2, 2, i+1)
    plt.plot(t_ms, true_filter, 'k-', linewidth=2, label='True')
    plt.plot(t_ms, filt, 'r-', linewidth=2, label=f'Reg={reg:.0e}')
    
    corr = np.corrcoef(true_filter, filt)[0, 1]
    plt.title(f'Regularization = {reg:.0e}\nCorrelation = {corr:.3f}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Filter Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Subspace Methods

```python
# Extract multiple filters using subspace methods
def extract_multiple_filters(stimulus, spikes, filter_length, n_filters=3):
    """Extract multiple filters using SVD of STC matrix"""
    
    # Create design matrix
    design_matrix = create_design_matrix_batch(stimulus, filter_length=filter_length)
    
    # Compute spike-triggered covariance (STC)
    spike_indices = np.where(spikes > 0)[0]
    
    if len(spike_indices) < 100:
        raise ValueError("Not enough spikes for subspace analysis")
    
    # Get spike-triggered stimuli
    spike_triggered_stimuli = design_matrix[spike_indices]
    
    # Compute covariance matrix
    stc_matrix = np.cov(spike_triggered_stimuli.T)
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(stc_matrix)
    
    # Extract top filters
    filters = Vt[:n_filters]
    eigenvalues = S[:n_filters]
    
    return filters, eigenvalues

# Example usage
try:
    multiple_filters, eigenvals = extract_multiple_filters(
        stimulus, spike_counts, filter_length, n_filters=3
    )
    
    plt.figure(figsize=(12, 4))
    t_ms = np.arange(filter_length) * 1
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(t_ms, multiple_filters[i], 'b-', linewidth=2)
        plt.title(f'Filter {i+1}\nEigenvalue: {eigenvals[i]:.3f}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except ValueError as e:
    print(f"Subspace analysis failed: {e}")
```

## Performance Optimization {#performance}

### Memory Profiling

```python
import psutil
import time

def profile_memory_usage(analyzer, stimulus, spikes, chunk_sizes):
    """Profile memory usage for different chunk sizes"""
    
    results = []
    
    for chunk_size in chunk_sizes:
        # Monitor memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**3  # GB
        
        # Create fresh analyzer
        test_analyzer = SingleCellAnalyzer(
            bin_size=0.001,
            filter_length=filter_length,
            memory_limit_gb=4.0
        )
        
        # Time the analysis
        start_time = time.time()
        
        try:
            stim_gen = create_stimulus_generator(stimulus, chunk_size=chunk_size)
            spike_gen = create_spike_generator(spikes, chunk_size=chunk_size)
            
            test_analyzer.fit_streaming(
                stim_gen, spike_gen,
                progress_bar=False
            )
            
            end_time = time.time()
            
            # Monitor memory after
            mem_after = process.memory_info().rss / 1024**3
            
            results.append({
                'chunk_size': chunk_size,
                'memory_usage_gb': mem_after - mem_before,
                'time_seconds': end_time - start_time,
                'success': True
            })
            
        except Exception as e:
            results.append({
                'chunk_size': chunk_size,
                'memory_usage_gb': None,
                'time_seconds': None,
                'success': False,
                'error': str(e)
            })
    
    return results

# Test different chunk sizes
chunk_sizes = [500, 1000, 2000, 5000, 10000]
profiling_results = profile_memory_usage(analyzer, stimulus, spike_counts, chunk_sizes)

# Plot results
successful_results = [r for r in profiling_results if r['success']]

if successful_results:
    chunk_sizes_success = [r['chunk_size'] for r in successful_results]
    memory_usage = [r['memory_usage_gb'] for r in successful_results]
    time_usage = [r['time_seconds'] for r in successful_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(chunk_sizes_success, memory_usage, 'bo-')
    ax1.set_xlabel('Chunk Size')
    ax1.set_ylabel('Memory Usage (GB)')
    ax1.set_title('Memory vs Chunk Size')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(chunk_sizes_success, time_usage, 'ro-')
    ax2.set_xlabel('Chunk Size')
    ax2.set_ylabel('Analysis Time (seconds)')
    ax2.set_title('Speed vs Chunk Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    optimal_idx = np.argmin(time_usage)
    print(f"Optimal chunk size for speed: {chunk_sizes_success[optimal_idx]}")
    print(f"Time: {time_usage[optimal_idx]:.2f}s, Memory: {memory_usage[optimal_idx]:.2f}GB")
```

### Parallel Processing Tips

```python
# For multi-electrode analysis, leverage parallel processing
from white_noise_toolkit import MultiElectrodeAnalyzer

def setup_parallel_analysis(n_electrodes=16):
    """Setup for parallel multi-electrode analysis"""
    
    # Generate multi-electrode data
    electrode_spikes = []
    
    for i in range(n_electrodes):
        # Each electrode has slightly different filter
        filter_variation = true_filter * (1 + 0.2 * np.random.randn())
        
        gen_i = SyntheticDataGenerator(
            filter_true=filter_variation,
            nonlinearity_true=lambda x: np.maximum(x + 0.3, 0),
            noise_level=0.1,
            random_seed=42 + i
        )
        
        spikes_i = gen_i.generate_responses(stimulus, bin_size=0.001)
        electrode_spikes.append(spikes_i)
    
    # Stack into multi-electrode format
    multi_electrode_spikes = np.column_stack(electrode_spikes)
    
    return multi_electrode_spikes

# Example parallel analysis setup
multi_spikes = setup_parallel_analysis(n_electrodes=8)
print(f"Multi-electrode data shape: {multi_spikes.shape}")

# Configure parallel analyzer
me_analyzer = MultiElectrodeAnalyzer(
    bin_size=0.001,
    filter_length=filter_length,
    n_jobs=4,  # Use 4 parallel processes
    min_spike_count=50
)

print("Multi-electrode analyzer configured for parallel processing")
```

## Statistical Validation {#validation}

### Bootstrap Confidence Intervals

```python
def bootstrap_filter_confidence(stimulus, spikes, n_bootstrap=100, confidence=0.95):
    """Compute bootstrap confidence intervals for filter estimates"""
    
    n_samples = len(stimulus)
    bootstrap_filters = []
    
    for i in range(n_bootstrap):
        # Bootstrap resample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        stim_boot = stimulus[indices]
        spikes_boot = spikes[indices]
        
        # Fit filter
        analyzer_boot = SingleCellAnalyzer(
            bin_size=0.001,
            filter_length=filter_length,
            memory_limit_gb=1.0
        )
        
        stim_gen_boot = create_stimulus_generator(stim_boot, chunk_size=1500)
        spike_gen_boot = create_spike_generator(spikes_boot, chunk_size=1500)
        
        try:
            analyzer_boot.fit_streaming(
                stim_gen_boot, spike_gen_boot,
                progress_bar=False
            )
            
            results_boot = analyzer_boot.get_results()
            bootstrap_filters.append(results_boot['filter'])
            
        except Exception:
            continue  # Skip failed bootstrap samples
    
    if len(bootstrap_filters) < 10:
        print("Warning: Few successful bootstrap samples")
        return None, None, None
    
    # Compute confidence intervals
    bootstrap_filters = np.array(bootstrap_filters)
    alpha = 1 - confidence
    
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_filters, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_filters, upper_percentile, axis=0)
    ci_median = np.median(bootstrap_filters, axis=0)
    
    return ci_lower, ci_upper, ci_median

# Compute confidence intervals (warning: this takes time!)
print("Computing bootstrap confidence intervals (this may take a minute)...")
ci_lower, ci_upper, ci_median = bootstrap_filter_confidence(
    stimulus[:10000], spike_counts[:10000],  # Use subset for speed
    n_bootstrap=50, confidence=0.95
)

if ci_lower is not None:
    # Plot confidence intervals
    t_ms = np.arange(filter_length) * 1
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(t_ms, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
    plt.plot(t_ms, ci_median, 'b-', linewidth=2, label='Bootstrap Median')
    plt.plot(t_ms, true_filter, 'k-', linewidth=2, label='True Filter')
    plt.plot(t_ms, estimated_filter, 'r--', linewidth=2, label='Original Estimate')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Filter Weight')
    plt.title('Bootstrap Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Real-World Examples {#examples}

### Example 1: Retinal Ganglion Cell Analysis

```python
def analyze_retinal_data():
    """Example analysis pipeline for retinal ganglion cell data"""
    
    # Simulate retinal-like parameters
    bin_size = 0.01  # 10ms bins (typical for retina)
    filter_length = 20  # 200ms filter
    
    # Retinal-like filter (biphasic)
    t = np.arange(filter_length) * bin_size
    retinal_filter = (np.exp(-t/0.05) - 0.3 * np.exp(-t/0.15)) * np.sin(2*np.pi*t/0.08)
    retinal_filter = retinal_filter / np.linalg.norm(retinal_filter)
    
    # Retinal-like nonlinearity (threshold-linear)
    def retinal_nonlinearity(x):
        threshold = 0.1
        slope = 100  # Hz per unit
        return np.maximum(slope * (x - threshold), 0)
    
    # Generate data
    generator = SyntheticDataGenerator(
        filter_true=retinal_filter,
        nonlinearity_true=retinal_nonlinearity,
        noise_level=0.08,
        random_seed=42
    )
    
    stimulus = generator.generate_white_noise_stimulus(30000)
    spikes = generator.generate_responses(stimulus, bin_size=bin_size)
    
    # Analysis
    analyzer = SingleCellAnalyzer(
        bin_size=bin_size,
        filter_length=filter_length,
        memory_limit_gb=2.0
    )
    
    stim_gen = create_stimulus_generator(stimulus, chunk_size=2000)
    spike_gen = create_spike_generator(spikes, chunk_size=2000)
    
    analyzer.fit_streaming(stim_gen, spike_gen)
    results = analyzer.get_results()
    
    return results, retinal_filter, retinal_nonlinearity

# Run retinal analysis
retinal_results, true_retinal_filter, true_retinal_nl = analyze_retinal_data()

# Visualize retinal-specific results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Biphasic filter
t_ms = np.arange(len(true_retinal_filter)) * 10  # 10ms bins
axes[0].plot(t_ms, true_retinal_filter, 'k-', linewidth=2, label='True')
axes[0].plot(t_ms, retinal_results['filter'], 'r-', linewidth=2, label='Estimated')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Filter Weight')
axes[0].set_title('Retinal Biphasic Filter')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Threshold nonlinearity
nl_data = retinal_results['nonlinearity']
x_vals = nl_data['x_values']
y_vals = nl_data['y_values']
x_fine = np.linspace(np.min(x_vals), np.max(x_vals), 100)
y_true = true_retinal_nl(x_fine)

axes[1].plot(x_fine, y_true, 'k-', linewidth=2, label='True')
axes[1].plot(x_vals, y_vals, 'ro-', markersize=4, label='Estimated')
axes[1].set_xlabel('Generator Signal')
axes[1].set_ylabel('Firing Rate (Hz)')
axes[1].set_title('Threshold-Linear Nonlinearity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Filter frequency response
freqs = np.fft.fftfreq(len(true_retinal_filter), 0.01)[:len(true_retinal_filter)//2]
true_fft = np.abs(np.fft.fft(true_retinal_filter))[:len(true_retinal_filter)//2]
est_fft = np.abs(np.fft.fft(retinal_results['filter']))[:len(true_retinal_filter)//2]

axes[2].loglog(freqs[1:], true_fft[1:], 'k-', linewidth=2, label='True')
axes[2].loglog(freqs[1:], est_fft[1:], 'r-', linewidth=2, label='Estimated')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Power')
axes[2].set_title('Filter Frequency Response')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Retinal ganglion cell analysis complete!")
```

## Summary

In this advanced tutorial, you learned:

1. **Parametric Models**: How to fit interpretable nonlinearity models
2. **Spatial-Temporal Analysis**: Handling 2D+time receptive fields
3. **Custom Nonlinearities**: Implementing specialized models
4. **Advanced Filters**: Regularization and subspace methods
5. **Performance Optimization**: Memory profiling and parallel processing
6. **Statistical Validation**: Bootstrap confidence intervals
7. **Real-World Examples**: Retinal ganglion cell analysis

### Next Steps

- **Tutorial 3**: Multi-electrode array analysis
- **Tutorial 4**: Large-scale data processing
- **Tutorial 5**: Integration with experimental workflows

The toolkit provides the flexibility to handle diverse neuroscience applications while maintaining computational efficiency and statistical rigor.

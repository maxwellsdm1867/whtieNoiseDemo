# Tutorial 3: Multi-Electrode Array Analysis

This tutorial demonstrates how to analyze data from multi-electrode arrays (MEAs) using the White Noise Analysis Toolkit's parallel processing capabilities.

## Table of Contents

1. [Introduction to MEA Analysis](#introduction)
2. [Data Formats and Organization](#data-formats)
3. [Parallel Processing Setup](#parallel-setup)
4. [Population Analysis](#population-analysis)
5. [Cross-Correlation Analysis](#cross-correlation)
6. [Visualization and Interpretation](#visualization)
7. [Performance Considerations](#performance)
8. [Real Experimental Workflow](#workflow)

## Introduction to MEA Analysis {#introduction}

Multi-electrode arrays enable simultaneous recording from many neurons, providing insights into population dynamics and functional connectivity. The toolkit handles:

- **Parallel Processing**: Simultaneous analysis of multiple electrodes
- **Population Statistics**: Cross-electrode comparisons and clustering
- **Memory Management**: Efficient handling of large multi-channel datasets
- **Quality Control**: Automatic filtering of low-activity electrodes

### Key Concepts

- **Electrode**: Individual recording site
- **Population Receptive Field**: Collection of all electrode filters
- **Functional Clustering**: Grouping electrodes by similar response properties
- **Cross-Correlation**: Temporal relationships between electrode responses

## Data Formats and Organization {#data-formats}

### Expected Data Structure

```python
import numpy as np
from white_noise_toolkit import MultiElectrodeAnalyzer, SyntheticDataGenerator

# Multi-electrode data format
# stimulus: (n_time_bins, *spatial_dims)  - shared across electrodes
# spikes: (n_time_bins, n_electrodes)     - electrode-specific responses

print("MEA Data Organization:")
print("  stimulus.shape = (n_time_bins,) or (n_time_bins, height, width)")
print("  spikes.shape = (n_time_bins, n_electrodes)")
print("  Each column in spikes corresponds to one electrode")
```

### Creating Synthetic MEA Data

```python
def create_mea_data(n_electrodes=16, n_samples=30000, spatial_layout=True):
    """Create synthetic MEA data with diverse electrode properties"""
    
    # Common stimulus for all electrodes
    bin_size = 0.008  # 8ms bins
    filter_length = 25
    
    if spatial_layout:
        # Spatial stimulus (simulating visual experiment)
        spatial_dims = (8, 8)
        stimulus_generator = SyntheticDataGenerator(
            filter_true=np.ones(filter_length),  # Dummy filter for stimulus generation
            nonlinearity_true=lambda x: x,
            random_seed=42
        )
        stimulus = stimulus_generator.generate_white_noise_stimulus(
            n_time_bins=n_samples,
            spatial_dims=spatial_dims
        )
    else:
        # Temporal stimulus (simulating auditory experiment)
        stimulus = np.random.randn(n_samples)
    
    # Create diverse electrode properties
    electrode_data = []
    electrode_metadata = []
    
    for i in range(n_electrodes):
        # Create unique filter for each electrode
        if spatial_layout:
            # Spatial filter with different preferred locations
            center_x = (i % 4) * 2 + 1  # Grid positions
            center_y = (i // 4) * 2 + 1
            
            # Create spatially localized filter
            filter_spatial = create_spatial_filter(
                spatial_dims, center_x, center_y, filter_length
            )
        else:
            # Temporal filter with different time constants
            t = np.arange(filter_length) * bin_size
            tau = 0.01 + 0.02 * (i / n_electrodes)  # Varying time constants
            freq = 50 + 100 * (i / n_electrodes)   # Varying frequencies
            
            filter_temporal = np.exp(-t/tau) * np.sin(2*np.pi*freq*t)
            filter_temporal = filter_temporal / np.linalg.norm(filter_temporal)
            filter_spatial = filter_temporal
        
        # Create unique nonlinearity
        threshold = -0.2 + 0.4 * (i / n_electrodes)  # Varying thresholds
        gain = 50 + 100 * np.random.rand()           # Varying gains
        
        def make_nonlinearity(thresh, g):
            return lambda x: np.maximum(g * (x - thresh), 0)
        
        nonlinearity = make_nonlinearity(threshold, gain)
        
        # Generate responses for this electrode
        electrode_gen = SyntheticDataGenerator(
            filter_true=filter_spatial,
            nonlinearity_true=nonlinearity,
            noise_level=0.1 + 0.1 * np.random.rand(),  # Varying noise
            random_seed=42 + i
        )
        
        electrode_spikes = electrode_gen.generate_responses(stimulus, bin_size=bin_size)
        electrode_data.append(electrode_spikes)
        
        # Store metadata
        electrode_metadata.append({
            'electrode_id': i,
            'true_filter': filter_spatial,
            'threshold': threshold,
            'gain': gain,
            'spike_count': np.sum(electrode_spikes),
            'firing_rate': np.sum(electrode_spikes) / (len(electrode_spikes) * bin_size)
        })
    
    # Stack into multi-electrode format
    mea_spikes = np.column_stack(electrode_data)
    
    return stimulus, mea_spikes, electrode_metadata

def create_spatial_filter(spatial_dims, center_x, center_y, filter_length):
    """Create spatially localized filter"""
    height, width = spatial_dims
    
    # Temporal component
    t = np.arange(filter_length) * 0.008
    temporal = np.exp(-t/0.025) * np.sin(2*np.pi*t/0.04)
    temporal = temporal / np.linalg.norm(temporal)
    
    # Spatial component (Gaussian)
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    sigma = 1.5
    spatial = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    spatial = spatial / np.linalg.norm(spatial)
    
    # Combine (separable)
    spatiotemporal = np.outer(temporal, spatial.ravel())
    return spatiotemporal.ravel()

# Generate MEA data
print("Generating synthetic MEA data...")
stimulus, mea_spikes, electrode_metadata = create_mea_data(
    n_electrodes=16, 
    n_samples=25000, 
    spatial_layout=False  # Use temporal for this example
)

print(f"Generated MEA data:")
print(f"  Stimulus shape: {stimulus.shape}")
print(f"  Spikes shape: {mea_spikes.shape}")
print(f"  Number of electrodes: {mea_spikes.shape[1]}")
print(f"  Total spikes: {np.sum(mea_spikes)}")

# Display electrode statistics
print("\nElectrode Statistics:")
for i, metadata in enumerate(electrode_metadata[:5]):  # Show first 5
    print(f"  Electrode {i}: {metadata['spike_count']} spikes, {metadata['firing_rate']:.1f} Hz")
```

## Parallel Processing Setup {#parallel-setup}

### Basic Multi-Electrode Analysis

```python
from white_noise_toolkit import MultiElectrodeAnalyzer

# Configure parallel analyzer
me_analyzer = MultiElectrodeAnalyzer(
    bin_size=0.008,
    filter_length=25,
    memory_limit_gb=4.0,
    n_jobs=4,  # Use 4 parallel processes
    min_spike_count=100,  # Exclude low-activity electrodes
    verbose=True
)

print("Multi-electrode analyzer configured:")
print(f"  Parallel processes: {me_analyzer.n_jobs}")
print(f"  Minimum spike count: {me_analyzer.min_spike_count}")
print(f"  Memory limit: {me_analyzer.memory_limit_gb} GB")
```

### Running Parallel Analysis

```python
# Method 1: Direct array input
results_mea = me_analyzer.analyze_electrode_array(
    stimulus=stimulus,
    spike_array=mea_spikes,
    electrode_ids=list(range(mea_spikes.shape[1])),
    chunk_size=2000,
    nonlinearity_method='nonparametric',
    progress_bar=True
)

print(f"\nAnalysis completed!")
print(f"Successfully analyzed: {len(results_mea['results'])} electrodes")
print(f"Failed electrodes: {len(results_mea['failed_electrodes'])}")

if results_mea['failed_electrodes']:
    print("Failed electrode details:")
    for electrode_id, error in results_mea['failed_electrodes'].items():
        print(f"  Electrode {electrode_id}: {error}")
```

### Advanced Parallel Configuration

```python
def setup_adaptive_parallel_processing():
    """Setup adaptive parallel processing based on system resources"""
    import psutil
    import multiprocessing
    
    # Detect system resources
    n_cpu = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Adaptive configuration
    if memory_gb > 16:
        # High-memory system
        n_jobs = min(n_cpu, 8)
        memory_per_job = 2.0
        chunk_size = 3000
    elif memory_gb > 8:
        # Medium-memory system
        n_jobs = min(n_cpu, 4)
        memory_per_job = 1.5
        chunk_size = 2000
    else:
        # Low-memory system
        n_jobs = min(n_cpu, 2)
        memory_per_job = 1.0
        chunk_size = 1000
    
    print(f"Adaptive configuration:")
    print(f"  Detected: {n_cpu} CPUs, {memory_gb:.1f} GB RAM")
    print(f"  Using: {n_jobs} parallel jobs")
    print(f"  Memory per job: {memory_per_job:.1f} GB")
    print(f"  Chunk size: {chunk_size}")
    
    return n_jobs, memory_per_job, chunk_size

# Apply adaptive configuration
n_jobs, memory_per_job, chunk_size = setup_adaptive_parallel_processing()

me_analyzer_adaptive = MultiElectrodeAnalyzer(
    bin_size=0.008,
    filter_length=25,
    memory_limit_gb=memory_per_job,
    n_jobs=n_jobs,
    min_spike_count=50
)
```

## Population Analysis {#population-analysis}

### Extracting Population Receptive Fields

```python
# Extract all successful results
successful_results = results_mea['results']
electrode_ids = list(successful_results.keys())

# Extract filters and nonlinearities
population_filters = []
population_nonlinearities = []
population_metadata = []

for electrode_id in electrode_ids:
    result = successful_results[electrode_id]
    
    population_filters.append(result['filter'])
    population_nonlinearities.append(result['nonlinearity'])
    
    # Compute filter statistics
    filter_stats = {
        'electrode_id': electrode_id,
        'filter_norm': np.linalg.norm(result['filter']),
        'filter_peak': np.max(np.abs(result['filter'])),
        'filter_peak_time': np.argmax(np.abs(result['filter'])) * 0.008,  # Convert to ms
        'snr_estimate': result.get('metadata', {}).get('filter_snr_estimate', None)
    }
    
    if 'performance_metrics' in result:
        perf = result['performance_metrics']
        filter_stats.update({
            'spike_count': perf.get('total_spikes', 0),
            'firing_rate': perf.get('spike_rate_hz', 0)
        })
    
    population_metadata.append(filter_stats)

population_filters = np.array(population_filters)
print(f"Population analysis:")
print(f"  {len(population_filters)} successful electrodes")
print(f"  Filter matrix shape: {population_filters.shape}")
```

### Population Statistics

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

def analyze_population_statistics(filters, metadata):
    """Compute population-level statistics"""
    
    # 1. Filter similarity matrix
    filter_distances = pdist(filters, metric='correlation')
    similarity_matrix = 1 - squareform(filter_distances)
    
    # 2. Population average filter
    pop_average_filter = np.mean(filters, axis=0)
    pop_std_filter = np.std(filters, axis=0)
    
    # 3. Temporal dynamics
    peak_times = [meta['filter_peak_time'] for meta in metadata]
    firing_rates = [meta['firing_rate'] for meta in metadata]
    
    # 4. Clustering analysis
    linkage_matrix = linkage(filter_distances, method='ward')
    
    return {
        'similarity_matrix': similarity_matrix,
        'population_average': pop_average_filter,
        'population_std': pop_std_filter,
        'peak_times': peak_times,
        'firing_rates': firing_rates,
        'linkage_matrix': linkage_matrix
    }

# Compute population statistics
pop_stats = analyze_population_statistics(population_filters, population_metadata)

# Visualize population properties
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Population average filter
t_ms = np.arange(len(pop_stats['population_average'])) * 8
axes[0, 0].fill_between(t_ms, 
                       pop_stats['population_average'] - pop_stats['population_std'],
                       pop_stats['population_average'] + pop_stats['population_std'],
                       alpha=0.3, color='blue', label='±1 STD')
axes[0, 0].plot(t_ms, pop_stats['population_average'], 'b-', linewidth=2, label='Population Average')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Filter Weight')
axes[0, 0].set_title('Population Average Filter')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Filter similarity matrix
im = axes[0, 1].imshow(pop_stats['similarity_matrix'], cmap='viridis', vmin=0, vmax=1)
axes[0, 1].set_xlabel('Electrode ID')
axes[0, 1].set_ylabel('Electrode ID')
axes[0, 1].set_title('Filter Similarity Matrix')
plt.colorbar(im, ax=axes[0, 1])

# 3. Peak time distribution
axes[0, 2].hist(pop_stats['peak_times'], bins=15, alpha=0.7, color='green')
axes[0, 2].set_xlabel('Peak Time (ms)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Filter Peak Time Distribution')
axes[0, 2].grid(True, alpha=0.3)

# 4. Firing rate distribution
axes[1, 0].hist(pop_stats['firing_rates'], bins=15, alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Firing Rate (Hz)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Population Firing Rate Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 5. Individual filters overlay
for i, filt in enumerate(population_filters[:10]):  # Show first 10
    alpha = 0.3 if i < 9 else 1.0  # Highlight last one
    linewidth = 1 if i < 9 else 2
    axes[1, 1].plot(t_ms, filt, alpha=alpha, linewidth=linewidth)
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Filter Weight')
axes[1, 1].set_title('Individual Electrode Filters')
axes[1, 1].grid(True, alpha=0.3)

# 6. Hierarchical clustering dendrogram
dendrogram(pop_stats['linkage_matrix'], ax=axes[1, 2], 
          labels=[f'E{i}' for i in electrode_ids])
axes[1, 2].set_xlabel('Electrode')
axes[1, 2].set_ylabel('Distance')
axes[1, 2].set_title('Hierarchical Clustering')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Functional Clustering

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def perform_functional_clustering(filters, n_clusters=3):
    """Cluster electrodes based on filter similarity"""
    
    # 1. Dimensionality reduction with PCA
    pca = PCA(n_components=min(10, filters.shape[1]))
    filters_pca = pca.fit_transform(filters)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative variance explained: {np.cumsum(pca.explained_variance_ratio_)[:5]}")
    
    # 2. K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(filters_pca)
    
    # 3. Analyze clusters
    cluster_info = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_electrodes = np.array(electrode_ids)[cluster_mask]
        cluster_filters = filters[cluster_mask]
        
        cluster_info[cluster_id] = {
            'electrodes': cluster_electrodes,
            'n_electrodes': np.sum(cluster_mask),
            'mean_filter': np.mean(cluster_filters, axis=0),
            'std_filter': np.std(cluster_filters, axis=0)
        }
    
    return cluster_labels, cluster_info, pca, filters_pca

# Perform clustering
cluster_labels, cluster_info, pca, filters_pca = perform_functional_clustering(
    population_filters, n_clusters=3
)

# Visualize clustering results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. PCA space with clusters
scatter = axes[0, 0].scatter(filters_pca[:, 0], filters_pca[:, 1], 
                           c=cluster_labels, cmap='viridis', s=60)
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
axes[0, 0].set_title('Electrode Clustering in PCA Space')
plt.colorbar(scatter, ax=axes[0, 0])

# Add electrode labels
for i, electrode_id in enumerate(electrode_ids):
    axes[0, 0].annotate(f'E{electrode_id}', 
                       (filters_pca[i, 0], filters_pca[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

# 2. Cluster mean filters
colors = ['red', 'blue', 'green', 'orange', 'purple']
for cluster_id, info in cluster_info.items():
    mean_filter = info['mean_filter']
    std_filter = info['std_filter']
    color = colors[cluster_id % len(colors)]
    
    axes[0, 1].fill_between(t_ms, mean_filter - std_filter, mean_filter + std_filter,
                           alpha=0.2, color=color)
    axes[0, 1].plot(t_ms, mean_filter, color=color, linewidth=2, 
                   label=f'Cluster {cluster_id} (n={info["n_electrodes"]})')

axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Filter Weight')
axes[0, 1].set_title('Cluster Mean Filters')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Cluster composition
cluster_sizes = [info['n_electrodes'] for info in cluster_info.values()]
axes[1, 0].pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(len(cluster_sizes))],
              autopct='%1.1f%%', colors=colors[:len(cluster_sizes)])
axes[1, 0].set_title('Cluster Size Distribution')

# 4. Firing rate by cluster
cluster_firing_rates = []
cluster_names = []
for cluster_id, info in cluster_info.items():
    cluster_electrodes = info['electrodes']
    rates = [metadata['firing_rate'] for metadata in population_metadata 
             if metadata['electrode_id'] in cluster_electrodes]
    cluster_firing_rates.extend(rates)
    cluster_names.extend([f'Cluster {cluster_id}'] * len(rates))

import pandas as pd
df_rates = pd.DataFrame({'cluster': cluster_names, 'firing_rate': cluster_firing_rates})

# Box plot
box_plot = axes[1, 1].boxplot([df_rates[df_rates['cluster'] == f'Cluster {i}']['firing_rate'].values 
                               for i in range(len(cluster_info))],
                              labels=[f'Cluster {i}' for i in range(len(cluster_info))])
axes[1, 1].set_ylabel('Firing Rate (Hz)')
axes[1, 1].set_title('Firing Rate by Cluster')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nClustering Summary:")
for cluster_id, info in cluster_info.items():
    print(f"Cluster {cluster_id}: {info['n_electrodes']} electrodes")
    print(f"  Electrodes: {info['electrodes']}")
```

## Cross-Correlation Analysis {#cross-correlation}

### Spike Train Cross-Correlation

```python
def compute_spike_cross_correlations(mea_spikes, max_lag_ms=50, bin_size_ms=8):
    """Compute cross-correlation between all electrode pairs"""
    
    n_electrodes = mea_spikes.shape[1]
    max_lag_bins = int(max_lag_ms / bin_size_ms)
    
    # Initialize cross-correlation matrix
    # Shape: (n_electrodes, n_electrodes, 2*max_lag_bins + 1)
    cross_corr_matrix = np.zeros((n_electrodes, n_electrodes, 2*max_lag_bins + 1))
    
    print(f"Computing cross-correlations for {n_electrodes} electrodes...")
    
    for i in range(n_electrodes):
        for j in range(n_electrodes):
            if i == j:
                # Auto-correlation
                cross_corr = np.correlate(mea_spikes[:, i], mea_spikes[:, i], mode='full')
            else:
                # Cross-correlation
                cross_corr = np.correlate(mea_spikes[:, i], mea_spikes[:, j], mode='full')
            
            # Extract central portion
            center = len(cross_corr) // 2
            start = center - max_lag_bins
            end = center + max_lag_bins + 1
            cross_corr_matrix[i, j, :] = cross_corr[start:end]
    
    # Normalize by spike counts
    spike_counts = np.sum(mea_spikes, axis=0)
    for i in range(n_electrodes):
        for j in range(n_electrodes):
            if spike_counts[i] > 0 and spike_counts[j] > 0:
                cross_corr_matrix[i, j, :] /= np.sqrt(spike_counts[i] * spike_counts[j])
    
    # Create lag axis
    lags_ms = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size_ms
    
    return cross_corr_matrix, lags_ms

# Compute cross-correlations
cross_corr_matrix, lags_ms = compute_spike_cross_correlations(mea_spikes)

print(f"Cross-correlation matrix shape: {cross_corr_matrix.shape}")
print(f"Lag range: {lags_ms[0]:.0f} to {lags_ms[-1]:.0f} ms")
```

### Analyzing Cross-Correlation Patterns

```python
def analyze_cross_correlation_patterns(cross_corr_matrix, lags_ms, electrode_ids):
    """Analyze cross-correlation patterns"""
    
    n_electrodes = len(electrode_ids)
    
    # 1. Zero-lag correlation matrix
    zero_lag_idx = len(lags_ms) // 2
    zero_lag_corr = cross_corr_matrix[:, :, zero_lag_idx]
    
    # 2. Peak correlation and lag for each pair
    peak_corr_matrix = np.zeros((n_electrodes, n_electrodes))
    peak_lag_matrix = np.zeros((n_electrodes, n_electrodes))
    
    for i in range(n_electrodes):
        for j in range(n_electrodes):
            cross_corr_trace = cross_corr_matrix[i, j, :]
            peak_idx = np.argmax(np.abs(cross_corr_trace))
            
            peak_corr_matrix[i, j] = cross_corr_trace[peak_idx]
            peak_lag_matrix[i, j] = lags_ms[peak_idx]
    
    # 3. Synchrony metrics
    synchrony_metrics = {
        'mean_zero_lag_corr': np.mean(zero_lag_corr[np.triu_indices(n_electrodes, k=1)]),
        'max_cross_corr': np.max(peak_corr_matrix[np.triu_indices(n_electrodes, k=1)]),
        'mean_peak_lag': np.mean(np.abs(peak_lag_matrix[np.triu_indices(n_electrodes, k=1)])),
        'n_strong_connections': np.sum(peak_corr_matrix[np.triu_indices(n_electrodes, k=1)] > 0.1)
    }
    
    return zero_lag_corr, peak_corr_matrix, peak_lag_matrix, synchrony_metrics

# Analyze cross-correlation patterns
zero_lag_corr, peak_corr_matrix, peak_lag_matrix, sync_metrics = analyze_cross_correlation_patterns(
    cross_corr_matrix, lags_ms, electrode_ids
)

print("Synchrony Metrics:")
for metric, value in sync_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

### Visualizing Cross-Correlation Results

```python
# Create comprehensive cross-correlation visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Zero-lag correlation matrix
im1 = axes[0, 0].imshow(zero_lag_corr, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0, 0].set_xlabel('Electrode ID')
axes[0, 0].set_ylabel('Electrode ID')
axes[0, 0].set_title('Zero-Lag Cross-Correlation')
plt.colorbar(im1, ax=axes[0, 0])

# 2. Peak correlation matrix
im2 = axes[0, 1].imshow(peak_corr_matrix, cmap='viridis', vmin=0, vmax=1)
axes[0, 1].set_xlabel('Electrode ID')
axes[0, 1].set_ylabel('Electrode ID')
axes[0, 1].set_title('Peak Cross-Correlation')
plt.colorbar(im2, ax=axes[0, 1])

# 3. Peak lag matrix
im3 = axes[0, 2].imshow(peak_lag_matrix, cmap='RdBu_r', vmin=-50, vmax=50)
axes[0, 2].set_xlabel('Electrode ID')
axes[0, 2].set_ylabel('Electrode ID')
axes[0, 2].set_title('Peak Lag (ms)')
plt.colorbar(im3, ax=axes[0, 2])

# 4. Example cross-correlation traces
example_pairs = [(0, 1), (0, 5), (2, 8)]  # Select interesting pairs
for pair_idx, (i, j) in enumerate(example_pairs):
    if i < len(electrode_ids) and j < len(electrode_ids):
        cross_corr_trace = cross_corr_matrix[i, j, :]
        axes[1, 0].plot(lags_ms, cross_corr_trace, 
                       label=f'E{electrode_ids[i]} → E{electrode_ids[j]}',
                       linewidth=2)

axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Lag (ms)')
axes[1, 0].set_ylabel('Cross-Correlation')
axes[1, 0].set_title('Example Cross-Correlation Traces')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Cross-correlation strength distribution
upper_tri_mask = np.triu_indices(len(electrode_ids), k=1)
cross_corr_strengths = peak_corr_matrix[upper_tri_mask]

axes[1, 1].hist(cross_corr_strengths, bins=20, alpha=0.7, color='purple')
axes[1, 1].axvline(sync_metrics['mean_zero_lag_corr'], color='red', 
                  linestyle='--', label='Mean Zero-Lag')
axes[1, 1].set_xlabel('Peak Cross-Correlation')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Cross-Correlation Strength Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Network connectivity graph
# Show strong connections as a network
import networkx as nx

# Create network from strong connections
threshold = 0.1  # Minimum correlation for connection
G = nx.Graph()

# Add nodes
for electrode_id in electrode_ids:
    G.add_node(electrode_id)

# Add edges for strong connections
for i, electrode_i in enumerate(electrode_ids):
    for j, electrode_j in enumerate(electrode_ids):
        if i < j and peak_corr_matrix[i, j] > threshold:
            G.add_edge(electrode_i, electrode_j, weight=peak_corr_matrix[i, j])

# Draw network
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, ax=axes[1, 2], node_color='lightblue', 
                      node_size=300)
nx.draw_networkx_labels(G, pos, ax=axes[1, 2], font_size=8)

# Draw edges with thickness proportional to correlation
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, ax=axes[1, 2], width=[w*5 for w in weights], 
                      alpha=0.6)

axes[1, 2].set_title(f'Functional Connectivity Network\n(threshold = {threshold})')
axes[1, 2].set_aspect('equal')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print(f"\nNetwork Analysis:")
print(f"  Nodes (electrodes): {G.number_of_nodes()}")
print(f"  Edges (connections): {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.3f}")
```

## Performance Considerations {#performance}

### Memory Optimization for Large Arrays

```python
def estimate_memory_requirements(n_electrodes, n_samples, filter_length):
    """Estimate memory requirements for MEA analysis"""
    
    # Data storage
    stimulus_mb = n_samples * 8 / (1024**2)  # 8 bytes per float64
    spikes_mb = n_samples * n_electrodes * 8 / (1024**2)
    
    # Analysis intermediate storage (per electrode)
    design_matrix_mb = n_samples * filter_length * 8 / (1024**2)
    correlation_matrices_mb = filter_length**2 * 8 / (1024**2)
    
    # Total per electrode
    per_electrode_mb = design_matrix_mb + correlation_matrices_mb
    
    # Total for all electrodes (if processed simultaneously)
    total_simultaneous_mb = stimulus_mb + spikes_mb + n_electrodes * per_electrode_mb
    
    # Sequential processing (recommended)
    total_sequential_mb = stimulus_mb + spikes_mb + per_electrode_mb
    
    return {
        'stimulus_mb': stimulus_mb,
        'spikes_mb': spikes_mb,
        'per_electrode_mb': per_electrode_mb,
        'total_simultaneous_gb': total_simultaneous_mb / 1024,
        'total_sequential_gb': total_sequential_mb / 1024
    }

# Estimate memory for our dataset
memory_est = estimate_memory_requirements(
    n_electrodes=mea_spikes.shape[1],
    n_samples=mea_spikes.shape[0],
    filter_length=25
)

print("Memory Requirements Estimate:")
for key, value in memory_est.items():
    if 'gb' in key:
        print(f"  {key}: {value:.2f} GB")
    else:
        print(f"  {key}: {value:.1f} MB")

print(f"\nRecommendation: Use sequential processing to limit memory to {memory_est['total_sequential_gb']:.1f} GB")
```

### Optimizing Chunk Size and Parallelization

```python
def benchmark_mea_analysis(stimulus, spikes, configurations):
    """Benchmark different MEA analysis configurations"""
    
    import time
    
    results = []
    
    for config in configurations:
        print(f"Testing configuration: {config}")
        
        try:
            # Create analyzer with this configuration
            analyzer = MultiElectrodeAnalyzer(
                bin_size=0.008,
                filter_length=25,
                memory_limit_gb=config['memory_limit_gb'],
                n_jobs=config['n_jobs'],
                min_spike_count=50
            )
            
            # Time the analysis
            start_time = time.time()
            
            # Use subset of electrodes for faster benchmarking
            n_test_electrodes = min(4, spikes.shape[1])
            test_spikes = spikes[:, :n_test_electrodes]
            
            test_results = analyzer.analyze_electrode_array(
                stimulus=stimulus,
                spike_array=test_spikes,
                electrode_ids=list(range(n_test_electrodes)),
                chunk_size=config['chunk_size'],
                progress_bar=False
            )
            
            end_time = time.time()
            
            # Record results
            results.append({
                'config': config,
                'time_seconds': end_time - start_time,
                'electrodes_analyzed': len(test_results['results']),
                'success_rate': len(test_results['results']) / n_test_electrodes,
                'memory_efficient': True
            })
            
        except Exception as e:
            results.append({
                'config': config,
                'time_seconds': None,
                'electrodes_analyzed': 0,
                'success_rate': 0,
                'error': str(e),
                'memory_efficient': False
            })
    
    return results

# Test different configurations
test_configurations = [
    {'n_jobs': 1, 'chunk_size': 1000, 'memory_limit_gb': 1.0},
    {'n_jobs': 2, 'chunk_size': 1500, 'memory_limit_gb': 1.5},
    {'n_jobs': 4, 'chunk_size': 2000, 'memory_limit_gb': 2.0},
]

print("Benchmarking MEA analysis configurations...")
benchmark_results = benchmark_mea_analysis(stimulus, mea_spikes, test_configurations)

print("\nBenchmark Results:")
for result in benchmark_results:
    config = result['config']
    if result['time_seconds'] is not None:
        print(f"  Config {config}: {result['time_seconds']:.1f}s, "
              f"{result['success_rate']:.0%} success rate")
    else:
        print(f"  Config {config}: FAILED - {result.get('error', 'Unknown error')}")
```

## Real Experimental Workflow {#workflow}

### Complete MEA Analysis Pipeline

```python
def complete_mea_pipeline(stimulus_file, spike_file, output_dir, config=None):
    """Complete MEA analysis pipeline for experimental data"""
    
    import os
    from white_noise_toolkit import load_data, save_data
    from datetime import datetime
    
    # Default configuration
    if config is None:
        config = {
            'bin_size': 0.008,
            'filter_length': 25,
            'n_jobs': 4,
            'memory_limit_gb': 2.0,
            'min_spike_count': 100,
            'chunk_size': 2000,
            'nonlinearity_method': 'nonparametric',
            'quality_control': True
        }
    
    print(f"Starting MEA analysis pipeline at {datetime.now()}")
    print(f"Configuration: {config}")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        stimulus_data = load_data(stimulus_file)
        spike_data = load_data(spike_file)
        
        stimulus = stimulus_data['stimulus']
        mea_spikes = spike_data['spikes']
        
        print(f"   Loaded stimulus: {stimulus.shape}")
        print(f"   Loaded spikes: {mea_spikes.shape}")
        
    except Exception as e:
        print(f"   ERROR loading data: {e}")
        return None
    
    # Step 2: Quality control
    print("\n2. Quality control...")
    if config['quality_control']:
        # Check electrode activity
        electrode_spike_counts = np.sum(mea_spikes, axis=0)
        active_electrodes = np.where(electrode_spike_counts >= config['min_spike_count'])[0]
        
        print(f"   Total electrodes: {mea_spikes.shape[1]}")
        print(f"   Active electrodes (>= {config['min_spike_count']} spikes): {len(active_electrodes)}")
        
        if len(active_electrodes) == 0:
            print("   ERROR: No electrodes meet minimum spike count")
            return None
        
        # Filter to active electrodes
        mea_spikes = mea_spikes[:, active_electrodes]
        electrode_ids = active_electrodes.tolist()
    else:
        electrode_ids = list(range(mea_spikes.shape[1]))
    
    # Step 3: Analysis
    print("\n3. Running MEA analysis...")
    analyzer = MultiElectrodeAnalyzer(
        bin_size=config['bin_size'],
        filter_length=config['filter_length'],
        n_jobs=config['n_jobs'],
        memory_limit_gb=config['memory_limit_gb'],
        min_spike_count=config['min_spike_count']
    )
    
    results = analyzer.analyze_electrode_array(
        stimulus=stimulus,
        spike_array=mea_spikes,
        electrode_ids=electrode_ids,
        chunk_size=config['chunk_size'],
        nonlinearity_method=config['nonlinearity_method'],
        progress_bar=True
    )
    
    print(f"   Successfully analyzed: {len(results['results'])} electrodes")
    print(f"   Failed: {len(results['failed_electrodes'])} electrodes")
    
    # Step 4: Population analysis
    print("\n4. Population analysis...")
    if len(results['results']) > 1:
        # Extract filters for population analysis
        successful_ids = list(results['results'].keys())
        filters = np.array([results['results'][eid]['filter'] for eid in successful_ids])
        
        # Clustering
        cluster_labels, cluster_info, _, _ = perform_functional_clustering(filters, n_clusters=3)
        
        # Cross-correlation
        successful_indices = [electrode_ids.index(eid) for eid in successful_ids]
        subset_spikes = mea_spikes[:, successful_indices]
        cross_corr_matrix, lags_ms = compute_spike_cross_correlations(subset_spikes)
        
        # Add to results
        results['population_analysis'] = {
            'cluster_labels': cluster_labels,
            'cluster_info': cluster_info,
            'cross_correlation_matrix': cross_corr_matrix,
            'cross_correlation_lags_ms': lags_ms
        }
    
    # Step 5: Save results
    print("\n5. Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'mea_analysis_{timestamp}.npz')
    save_data(results, results_file)
    
    # Save configuration
    config_file = os.path.join(output_dir, f'config_{timestamp}.yaml')
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    print(f"   Results saved to: {results_file}")
    print(f"   Configuration saved to: {config_file}")
    
    print(f"\nMEA analysis pipeline completed at {datetime.now()}")
    return results

# Example usage (commented out since we don't have real files)
# results = complete_mea_pipeline(
#     stimulus_file='path/to/stimulus.h5',
#     spike_file='path/to/spikes.h5',
#     output_dir='./mea_analysis_output',
#     config={
#         'bin_size': 0.008,
#         'filter_length': 25,
#         'n_jobs': 4,
#         'memory_limit_gb': 3.0,
#         'min_spike_count': 100,
#         'chunk_size': 2500
#     }
# )

print("Complete MEA analysis pipeline defined and ready to use!")
```

## Summary

This tutorial covered comprehensive multi-electrode array analysis:

1. **Data Organization**: Proper formatting and synthetic data generation
2. **Parallel Processing**: Efficient multi-electrode analysis setup
3. **Population Analysis**: Statistical analysis of electrode populations
4. **Functional Clustering**: Grouping electrodes by response similarity
5. **Cross-Correlation**: Analyzing temporal relationships between electrodes
6. **Performance Optimization**: Memory and computational efficiency
7. **Real Workflow**: Complete experimental data analysis pipeline

### Key Takeaways

- **Parallel Processing**: Essential for large electrode arrays
- **Memory Management**: Critical for avoiding system limits
- **Population Perspective**: Reveals functional organization
- **Quality Control**: Important for reliable results
- **Comprehensive Analysis**: Combines single-cell and network approaches

### Next Steps

- **Tutorial 4**: Large-scale data processing and optimization
- **Tutorial 5**: Integration with experimental acquisition systems
- **Advanced Topics**: Real-time analysis and closed-loop experiments

The toolkit's MEA capabilities enable sophisticated population-level neuroscience research while maintaining computational efficiency.

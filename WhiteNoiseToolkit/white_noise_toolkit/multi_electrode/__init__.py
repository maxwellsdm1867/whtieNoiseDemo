"""
Multi-electrode analysis module for white noise experiments.

This module provides classes and functions for analyzing white noise responses
from multi-electrode arrays (MEAs) and analyzing population responses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

from ..core.single_cell import SingleCellAnalyzer
from ..core.exceptions import InsufficientDataError, DataValidationError
from ..utils.memory_manager import MemoryManager
from ..utils.logging_config import get_logger, TimingLogger

logger = get_logger(__name__)


class MultiElectrodeAnalyzer:
    """
    Analyzer for multi-electrode array data.

    Handles analysis of multiple neurons simultaneously with shared stimulus,
    including population analysis and cross-correlations.
    """

    def __init__(self,
                 bin_size: float = 0.008,
                 filter_length: int = 25,
                 spatial_dims: Optional[Tuple[int, ...]] = None,
                 n_colors: int = 1,
                 memory_limit_gb: float = 4.0,
                 n_workers: Optional[int] = None):
        """
        Initialize multi-electrode analyzer.

        Parameters
        ----------
        bin_size : float
            Temporal bin size in seconds
        filter_length : int
            Number of time bins for filter
        spatial_dims : tuple, optional
            (height, width) for spatial stimuli
        n_colors : int
            Number of color channels
        memory_limit_gb : float
            Memory limit for analysis (GB)
        n_workers : int, optional
            Number of worker processes. If None, uses CPU count
        """
        self.bin_size = bin_size
        self.filter_length = filter_length
        self.spatial_dims = spatial_dims
        self.n_colors = n_colors
        self.memory_limit_gb = memory_limit_gb
        self.n_workers = n_workers

        self.memory_manager = MemoryManager(memory_limit_gb)

        # Storage for individual analyzers
        self.analyzers: Dict[str, SingleCellAnalyzer] = {}
        self.results: Dict[str, Any] = {}

        # Population analysis results
        self.population_results: Dict[str, Any] = {}

        logger.info(f"Initialized MultiElectrodeAnalyzer with {n_workers} workers")

    def add_unit(self,
                 unit_id: str,
                 spike_times: np.ndarray,
                 **kwargs) -> None:
        """
        Add a unit for analysis.

        Parameters
        ----------
        unit_id : str
            Unique identifier for the unit
        spike_times : numpy.ndarray
            Spike times for this unit
        **kwargs
            Additional arguments for SingleCellAnalyzer
        """
        # Validate spike times
        if len(spike_times) == 0:
            logger.warning(f"Unit {unit_id} has no spikes, skipping")
            return

        # Create analyzer for this unit
        analyzer = SingleCellAnalyzer(
            bin_size=self.bin_size,
            filter_length=self.filter_length,
            spatial_dims=self.spatial_dims,
            n_colors=self.n_colors,
            memory_limit_gb=self.memory_limit_gb / len(self.analyzers) if self.analyzers else self.memory_limit_gb
        )
        self.analyzers[unit_id] = analyzer

        logger.info(f"Added unit {unit_id} with {len(spike_times)} spikes")

    def analyze_population(self,
                          stimulus: np.ndarray,
                          sampling_rate: float,
                          all_spike_times: Dict[str, np.ndarray],
                          parallel: bool = True) -> Dict[str, Any]:
        """
        Analyze all units in the population.

        Parameters
        ----------
        stimulus : numpy.ndarray
            Shared stimulus for all units
        sampling_rate : float
            Stimulus sampling rate
        all_spike_times : dict
            Dictionary mapping unit_id to spike times
        parallel : bool
            Whether to use parallel processing

        Returns
        -------
        dict
            Population analysis results
        """
        with TimingLogger(logger, "Population analysis"):
            # Add all units if not already added
            for unit_id, spike_times in all_spike_times.items():
                if unit_id not in self.analyzers:
                    self.add_unit(unit_id, spike_times)

            # Run individual analyses
            if parallel and len(self.analyzers) > 1:
                self._analyze_parallel(stimulus, sampling_rate, all_spike_times)
            else:
                self._analyze_sequential(stimulus, sampling_rate, all_spike_times)

            # Compute population-level metrics
            self._compute_population_metrics()

            # Cross-correlation analysis
            self._compute_cross_correlations(all_spike_times)

            return self.get_population_results()

    def _run_single_cell_analysis(self, analyzer: SingleCellAnalyzer,
                                 stimulus: np.ndarray, spike_times: np.ndarray,
                                 sampling_rate: float) -> Dict[str, Any]:
        """
        Run single cell analysis using the appropriate interface.

        Parameters
        ----------
        analyzer : SingleCellAnalyzer
            The analyzer instance
        stimulus : numpy.ndarray
            Stimulus data
        spike_times : numpy.ndarray
            Spike times
        sampling_rate : float
            Sampling rate

        Returns
        -------
        dict
            Analysis results
        """
        from ..core.streaming_analyzer import create_stimulus_generator, create_spike_generator

        # Convert spike times to binned format
        # Create time bins aligned with stimulus
        dt = 1.0 / sampling_rate
        n_bins = len(stimulus)
        time_edges = np.arange(n_bins + 1) * dt

        # Bin the spikes
        spike_counts, _ = np.histogram(spike_times, bins=time_edges)

        # Create generators for streaming analysis
        chunk_size = 1000
        stimulus_gen = create_stimulus_generator(stimulus, chunk_size)
        spike_gen = create_spike_generator(spike_counts, chunk_size)

        # Run the analysis
        analyzer.fit_streaming(stimulus_gen, spike_gen, chunk_size=chunk_size)

        # Return results
        return analyzer.get_results()

    def _analyze_sequential(self,
                           stimulus: np.ndarray,
                           sampling_rate: float,
                           all_spike_times: Dict[str, np.ndarray]) -> None:
        """Sequential analysis of all units."""
        for unit_id, spike_times in all_spike_times.items():
            if unit_id in self.analyzers:
                try:
                    logger.info(f"Analyzing unit {unit_id}")
                    result = self._run_single_cell_analysis(
                        self.analyzers[unit_id], stimulus, spike_times, sampling_rate)
                    self.results[unit_id] = result
                except Exception as e:
                    logger.error(f"Failed to analyze unit {unit_id}: {str(e)}")
                    self.results[unit_id] = None

    def _analyze_parallel(self,
                         stimulus: np.ndarray,
                         sampling_rate: float,
                         all_spike_times: Dict[str, np.ndarray]) -> None:
        """Parallel analysis of all units."""
        def analyze_unit(unit_data):
            unit_id, spike_times = unit_data
            try:
                analyzer = self.analyzers[unit_id]
                result = self._run_single_cell_analysis(
                    analyzer, stimulus, spike_times, sampling_rate)
                return unit_id, result
            except Exception as e:
                logger.error(f"Failed to analyze unit {unit_id}: {str(e)}")
                return unit_id, None

        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            unit_items = [(uid, spikes) for uid, spikes in all_spike_times.items()
                         if uid in self.analyzers]

            results = list(executor.map(analyze_unit, unit_items))

            for unit_id, result in results:
                self.results[unit_id] = result

    def _compute_population_metrics(self) -> None:
        """Compute population-level metrics."""
        valid_results = {uid: res for uid, res in self.results.items()
                        if res is not None}

        if not valid_results:
            logger.warning("No valid unit results for population analysis")
            return

        # Extract filters and nonlinearities
        filters = {}
        nonlinearities = {}
        filter_qualities = {}

        for unit_id, result in valid_results.items():
            if 'filter' in result:
                filters[unit_id] = result['filter']
            if 'nonlinearity' in result:
                nonlinearities[unit_id] = result['nonlinearity']
            if 'filter_quality' in result:
                filter_qualities[unit_id] = result['filter_quality']

        # Population filter analysis
        if filters:
            self.population_results['filter_analysis'] = self._analyze_population_filters(filters)

        # Population nonlinearity analysis
        if nonlinearities:
            self.population_results['nonlinearity_analysis'] = self._analyze_population_nonlinearities(nonlinearities)

        # Quality metrics
        if filter_qualities:
            self.population_results['quality_summary'] = self._summarize_quality_metrics(filter_qualities)

        # Count statistics
        self.population_results['unit_count'] = len(valid_results)
        self.population_results['successful_analyses'] = len(valid_results)
        self.population_results['failed_analyses'] = len(self.results) - len(valid_results)

        logger.info(f"Population analysis completed for {len(valid_results)} units")

    def _analyze_population_filters(self, filters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze population of filters."""
        if not filters:
            return {}

        # Stack all filters (pad to same length if needed)
        max_len = max(len(f) for f in filters.values())
        filter_matrix = np.zeros((len(filters), max_len))

        for i, (unit_id, filt) in enumerate(filters.items()):
            if len(filt) < max_len:
                # Pad with zeros
                padded_filter = np.zeros(max_len)
                padded_filter[:len(filt)] = filt
                filter_matrix[i] = padded_filter
            else:
                filter_matrix[i] = filt[:max_len]

        # Compute population statistics
        mean_filter = np.mean(filter_matrix, axis=0)
        std_filter = np.std(filter_matrix, axis=0)

        # Principal component analysis
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(5, len(filters)))
            pca_components = pca.fit_transform(filter_matrix)

            pca_results = {
                'components': pca.components_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'transformed_filters': pca_components
            }
        except ImportError:
            logger.warning("sklearn not available, skipping PCA analysis")
            pca_results = None

        # Clustering analysis (simple k-means)
        cluster_results = self._cluster_filters(filter_matrix)

        return {
            'mean_filter': mean_filter,
            'std_filter': std_filter,
            'filter_matrix': filter_matrix,
            'pca': pca_results,
            'clusters': cluster_results,
            'unit_ids': list(filters.keys())
        }

    def _cluster_filters(self, filter_matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        """Cluster filters using k-means."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            # Try different numbers of clusters
            n_units = filter_matrix.shape[0]
            max_clusters = min(5, n_units - 1)

            if max_clusters < 2:
                return None

            best_k = 2
            best_score = -1

            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(filter_matrix)

                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(filter_matrix, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

            # Final clustering with best k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(filter_matrix)

            return {
                'n_clusters': best_k,
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'silhouette_score': best_score
            }

        except ImportError:
            logger.warning("sklearn not available, skipping clustering analysis")
            return None

    def _analyze_population_nonlinearities(self, nonlinearities: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze population of nonlinearities."""
        if not nonlinearities:
            return {}

        # Extract x and y values for each unit
        x_values = {}
        y_values = {}

        for unit_id, nl_data in nonlinearities.items():
            if 'x_values' in nl_data and 'y_values' in nl_data:
                x_values[unit_id] = nl_data['x_values']
                y_values[unit_id] = nl_data['y_values']

        if not x_values:
            return {}

        # Find common x range
        min_x = max(np.min(x) for x in x_values.values())
        max_x = min(np.max(x) for x in x_values.values())

        # Interpolate all nonlinearities to common grid
        common_x = np.linspace(min_x, max_x, 100)
        interpolated_y = {}

        for unit_id in x_values.keys():
            x = x_values[unit_id]
            y = y_values[unit_id]

            # Sort by x values
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]

            # Interpolate
            interpolated_y[unit_id] = np.interp(common_x, x_sorted, y_sorted)

        # Stack interpolated nonlinearities
        nl_matrix = np.array([interpolated_y[uid] for uid in interpolated_y.keys()])

        # Compute population statistics
        mean_nonlinearity = np.mean(nl_matrix, axis=0)
        std_nonlinearity = np.std(nl_matrix, axis=0)

        # Compute diversity metrics
        pairwise_correlations = np.corrcoef(nl_matrix)
        mean_correlation = np.mean(pairwise_correlations[np.triu_indices_from(pairwise_correlations, k=1)])

        return {
            'common_x': common_x,
            'mean_nonlinearity': mean_nonlinearity,
            'std_nonlinearity': std_nonlinearity,
            'nonlinearity_matrix': nl_matrix,
            'pairwise_correlations': pairwise_correlations,
            'mean_correlation': float(mean_correlation),
            'unit_ids': list(interpolated_y.keys())
        }

    def _summarize_quality_metrics(self, qualities: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize quality metrics across units."""
        # Extract common metrics
        snr_values = []
        r2_values = []

        for unit_id, quality in qualities.items():
            if isinstance(quality, dict):
                if 'snr' in quality:
                    snr_values.append(quality['snr'])
                if 'r2' in quality:
                    r2_values.append(quality['r2'])

        summary = {}

        if snr_values:
            summary['snr'] = {
                'mean': float(np.mean(snr_values)),
                'std': float(np.std(snr_values)),
                'min': float(np.min(snr_values)),
                'max': float(np.max(snr_values)),
                'values': snr_values
            }

        if r2_values:
            summary['r2'] = {
                'mean': float(np.mean(r2_values)),
                'std': float(np.std(r2_values)),
                'min': float(np.min(r2_values)),
                'max': float(np.max(r2_values)),
                'values': r2_values
            }

        return summary

    def _compute_cross_correlations(self, spike_times: Dict[str, np.ndarray]) -> None:
        """Compute cross-correlations between units."""
        unit_ids = list(spike_times.keys())
        n_units = len(unit_ids)

        if n_units < 2:
            return

        cross_corr_matrix = np.zeros((n_units, n_units))
        max_lag = 0.1  # 100ms max lag

        for i, unit_i in enumerate(unit_ids):
            for j, unit_j in enumerate(unit_ids):
                if i <= j:  # Only compute upper triangle
                    if i == j:
                        cross_corr_matrix[i, j] = 1.0  # Auto-correlation peak
                    else:
                        # Compute cross-correlation
                        corr = self._compute_spike_cross_correlation(
                            spike_times[unit_i], spike_times[unit_j], max_lag)
                        cross_corr_matrix[i, j] = corr
                        cross_corr_matrix[j, i] = corr  # Symmetric

        self.population_results['cross_correlations'] = {
            'matrix': cross_corr_matrix,
            'unit_ids': unit_ids,
            'max_lag': max_lag
        }

    def _compute_spike_cross_correlation(self,
                                       spikes1: np.ndarray,
                                       spikes2: np.ndarray,
                                       max_lag: float) -> float:
        """Compute peak cross-correlation between two spike trains."""
        if len(spikes1) == 0 or len(spikes2) == 0:
            return 0.0

        # Create binary spike trains
        dt = 0.001  # 1ms resolution

        # Find common time range
        min_time = min(np.min(spikes1), np.min(spikes2))
        max_time = max(np.max(spikes1), np.max(spikes2))

        time_bins = np.arange(min_time, max_time + dt, dt)

        # Convert to binary
        binary1 = np.histogram(spikes1, bins=time_bins)[0]
        binary2 = np.histogram(spikes2, bins=time_bins)[0]

        # Compute cross-correlation
        correlation = np.correlate(binary1, binary2, mode='full')

        # Find peak within max_lag
        center = len(correlation) // 2
        lag_samples = int(max_lag / dt)

        start_idx = max(0, center - lag_samples)
        end_idx = min(len(correlation), center + lag_samples + 1)

        peak_corr = np.max(correlation[start_idx:end_idx])

        # Normalize by geometric mean of firing rates
        norm_factor = np.sqrt(np.sum(binary1) * np.sum(binary2))

        if norm_factor > 0:
            return float(peak_corr / norm_factor)
        else:
            return 0.0

    def get_unit_results(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific unit."""
        return self.results.get(unit_id)

    def get_population_results(self) -> Dict[str, Any]:
        """Get population-level results."""
        return {
            'individual_results': self.results,
            'population_analysis': self.population_results,
            'summary': {
                'n_units': len(self.analyzers),
                'successful_analyses': len([r for r in self.results.values() if r is not None]),
                'failed_analyses': len([r for r in self.results.values() if r is None])
            }
        }

    def get_successful_units(self) -> List[str]:
        """Get list of units with successful analyses."""
        return [uid for uid, result in self.results.items() if result is not None]

    def save_results(self, filepath: str) -> None:
        """Save all results to file."""
        from ..utils.io_handlers import save_data

        all_results = self.get_population_results()
        save_data(all_results, filepath)
        logger.info(f"Results saved to {filepath}")


def analyze_mea_data(stimulus: np.ndarray,
                    spike_data: Dict[str, np.ndarray],
                    sampling_rate: float,
                    bin_size: float = 0.008,
                    filter_length: int = 25,
                    spatial_dims: Optional[Tuple[int, ...]] = None,
                    n_colors: int = 1,
                    memory_limit_gb: float = 4.0,
                    parallel: bool = True,
                    n_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function for analyzing MEA data.

    Parameters
    ----------
    stimulus : numpy.ndarray
        Stimulus data
    spike_data : dict
        Dictionary mapping unit_id to spike times
    sampling_rate : float
        Stimulus sampling rate
    bin_size : float
        Temporal bin size in seconds
    filter_length : int
        Number of time bins for filter
    spatial_dims : tuple, optional
        (height, width) for spatial stimuli
    n_colors : int
        Number of color channels
    memory_limit_gb : float
        Memory limit for analysis (GB)
    parallel : bool
        Whether to use parallel processing
    n_workers : int, optional
        Number of worker processes

    Returns
    -------
    dict
        Analysis results
    """
    analyzer = MultiElectrodeAnalyzer(
        bin_size=bin_size,
        filter_length=filter_length,
        spatial_dims=spatial_dims,
        n_colors=n_colors,
        memory_limit_gb=memory_limit_gb,
        n_workers=n_workers
    )

    return analyzer.analyze_population(
        stimulus, sampling_rate, spike_data, parallel=parallel
    )

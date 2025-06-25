"""
Validation utilities for the White Noise Analysis Toolkit.

This module provides utilities for validating analysis results,
testing ground truth recovery, and benchmarking performance.
"""

import warnings
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from ..synthetic.data_generator import SyntheticDataGenerator
from ..core.single_cell import SingleCellAnalyzer
from ..utils.metrics import FilterMetrics, NonlinearityMetrics, ModelMetrics
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class GroundTruthRecovery:
    """
    Validate analysis by testing recovery of known ground truth.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize ground truth recovery validator.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def test_filter_recovery(self,
                           true_filter: np.ndarray,
                           stimulus_length: int = 10000,
                           noise_level: float = 0.1,
                           test_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test recovery of a known linear filter.

        Parameters
        ----------
        true_filter : numpy.ndarray
            Ground truth filter
        stimulus_length : int
            Length of stimulus to generate
        noise_level : float
            Noise level for spike generation
        test_methods : list, optional
            Analysis methods to test

        Returns
        -------
        dict
            Recovery test results
        """
        if test_methods is None:
            test_methods = ['STA', 'whitened_STA']

        # Create synthetic data generator
        def exponential_nonlinearity(x):
            return np.exp(x)

        generator = SyntheticDataGenerator(
            filter_true=true_filter,
            nonlinearity_true=exponential_nonlinearity,
            noise_level=noise_level,
            random_seed=self.random_state
        )

        # Generate synthetic data
        stimulus = generator.generate_white_noise_stimulus(stimulus_length)
        spike_counts = generator.generate_responses(stimulus, bin_size=0.001)

        # Convert spike counts to spike times (for compatibility)
        spike_times = []
        for i, count in enumerate(spike_counts):
            if count > 0:
                # Distribute spikes randomly within the bin
                bin_start = i * 0.001
                bin_spikes = np.random.uniform(bin_start, bin_start + 0.001, int(count))
                spike_times.extend(bin_spikes)
        spike_times = np.sort(spike_times)

        # Run analysis
        analyzer = SingleCellAnalyzer(
            bin_size=0.001,
            filter_length=len(true_filter)
        )

        from ..core.streaming_analyzer import create_stimulus_generator, create_spike_generator

        # Convert spike times to binned format
        dt = 0.001
        n_bins = len(stimulus)
        time_edges = np.arange(n_bins + 1) * dt
        spike_counts, _ = np.histogram(spike_times, bins=time_edges)

        # Create generators
        stimulus_gen = create_stimulus_generator(stimulus, chunk_size=1000)
        spike_gen = create_spike_generator(spike_counts, chunk_size=1000)

        # Run analysis
        analyzer.fit_streaming(stimulus_gen, spike_gen)
        results = analyzer.get_results()

        # Evaluate recovery quality
        recovery_metrics = {}

        for method in test_methods:
            if method == 'STA' and 'sta' in results:
                estimated_filter = results['sta']
            elif method == 'whitened_STA' and 'filter' in results:
                estimated_filter = results['filter']
            else:
                continue

            # Align filters (handle sign flip and delay)
            aligned_estimate, alignment_info = self._align_filters(
                true_filter, estimated_filter)

            # Compute recovery metrics
            metrics = {
                'correlation': FilterMetrics.filter_consistency(
                    true_filter, aligned_estimate, 'correlation'),
                'rmse': FilterMetrics.filter_consistency(
                    true_filter, aligned_estimate, 'rmse'),
                'snr': FilterMetrics.signal_to_noise_ratio(aligned_estimate),
                'alignment': alignment_info
            }

            recovery_metrics[method] = metrics

        return {
            'recovery_metrics': recovery_metrics,
            'true_filter': true_filter,
            'estimated_filters': {method: results.get('sta' if method == 'STA' else 'filter')
                                 for method in test_methods},
            'stimulus_stats': {
                'length': len(stimulus),
                'std': np.std(stimulus),
                'mean': np.mean(stimulus)
            },
            'spike_stats': {
                'count': len(spike_times),
                'rate': len(spike_times) / (stimulus_length * 0.001),
                'cv': self._compute_cv(spike_times) if len(spike_times) > 1 else np.nan
            }
        }

    def test_nonlinearity_recovery(self,
                                 nonlinearity_params: Dict[str, Any],
                                 filter_length: int = 20,
                                 stimulus_length: int = 20000) -> Dict[str, Any]:
        """
        Test recovery of a known nonlinearity.

        Parameters
        ----------
        nonlinearity_params : dict
            Parameters defining the true nonlinearity
        filter_length : int
            Length of filter to use
        stimulus_length : int
            Length of stimulus

        Returns
        -------
        dict
            Nonlinearity recovery results
        """
        # Create a simple temporal filter
        t = np.arange(filter_length) * 0.001
        true_filter = np.exp(-t/0.01) * np.sin(2*np.pi*t/0.005)
        true_filter = true_filter / np.linalg.norm(true_filter)

        # Create nonlinearity function based on parameters
        nl_type = nonlinearity_params.get('type', 'exponential')
        if nl_type == 'exponential':
            def nonlinearity_func(x):
                threshold = nonlinearity_params.get('threshold', 0.0)
                slope = nonlinearity_params.get('slope', 1.0)
                return np.exp(slope * np.maximum(x - threshold, 0))
        else:
            def nonlinearity_func(x):
                return np.maximum(x, 0)  # ReLU as fallback

        # Create synthetic data
        generator = SyntheticDataGenerator(
            filter_true=true_filter,
            nonlinearity_true=nonlinearity_func,
            noise_level=0.1,
            random_seed=self.random_state
        )

        stimulus = generator.generate_white_noise_stimulus(stimulus_length)
        spike_counts = generator.generate_responses(stimulus, bin_size=0.001)

        # Convert to spike times
        spike_times = []
        for i, count in enumerate(spike_counts):
            if count > 0:
                bin_start = i * 0.001
                bin_spikes = np.random.uniform(bin_start, bin_start + 0.001, int(count))
                spike_times.extend(bin_spikes)
        spike_times = np.sort(spike_times)

        # Run analysis with focus on nonlinearity
        analyzer = SingleCellAnalyzer(
            bin_size=0.001,
            filter_length=filter_length
        )

        from ..core.streaming_analyzer import create_stimulus_generator, create_spike_generator

        # Convert to binned format
        dt = 0.001
        n_bins = len(stimulus)
        time_edges = np.arange(n_bins + 1) * dt
        spike_counts, _ = np.histogram(spike_times, bins=time_edges)

        # Run analysis
        stimulus_gen = create_stimulus_generator(stimulus, 1000)
        spike_gen = create_spike_generator(spike_counts, 1000)

        analyzer.fit_streaming(stimulus_gen, spike_gen,
                             nonlinearity_method='nonparametric')
        results = analyzer.get_results()

        # Evaluate nonlinearity recovery
        if 'nonlinearity' not in results:
            return {'error': 'Nonlinearity estimation failed'}

        nl_data = results['nonlinearity']

        # Generate true nonlinearity for comparison using the same function
        x_range = np.linspace(
            np.min(nl_data['x_values']),
            np.max(nl_data['x_values']),
            100
        )
        true_nl = nonlinearity_func(x_range)
        estimated_nl = np.interp(x_range, nl_data['x_values'], nl_data['y_values'])

        # Compute recovery metrics
        recovery_metrics = {
            'r_squared': NonlinearityMetrics.r_squared(true_nl, estimated_nl),
            'correlation': np.corrcoef(true_nl, estimated_nl)[0, 1],
            'rmse': np.sqrt(np.mean((true_nl - estimated_nl) ** 2)),
            'nonlinearity_strength_true': NonlinearityMetrics.nonlinearity_strength(
                x_range, true_nl),
            'nonlinearity_strength_estimated': NonlinearityMetrics.nonlinearity_strength(
                x_range, estimated_nl)
        }

        return {
            'recovery_metrics': recovery_metrics,
            'true_nonlinearity': {'x': x_range, 'y': true_nl},
            'estimated_nonlinearity': nl_data,
            'nonlinearity_params': nonlinearity_params
        }

    def _align_filters(self, true_filter: np.ndarray,
                      estimated_filter: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Align estimated filter with true filter.

        Handles sign flips and time delays.
        """
        # Ensure same length
        min_len = min(len(true_filter), len(estimated_filter))
        true_filter = true_filter[:min_len]
        estimated_filter = estimated_filter[:min_len]

        # Try both polarities
        corr_pos = np.corrcoef(true_filter, estimated_filter)[0, 1]
        corr_neg = np.corrcoef(true_filter, -estimated_filter)[0, 1]

        if np.abs(corr_neg) > np.abs(corr_pos):
            aligned_filter = -estimated_filter
            sign_flip = True
            correlation = corr_neg
        else:
            aligned_filter = estimated_filter
            sign_flip = False
            correlation = corr_pos

        # Find optimal delay using cross-correlation
        cross_corr = np.correlate(true_filter, aligned_filter, mode='full')
        delay = np.argmax(cross_corr) - (len(aligned_filter) - 1)

        # Apply delay correction if reasonable
        if abs(delay) < len(aligned_filter) // 2:
            if delay > 0:
                aligned_filter = np.roll(aligned_filter, delay)
                aligned_filter[:delay] = 0
            elif delay < 0:
                aligned_filter = np.roll(aligned_filter, delay)
                aligned_filter[delay:] = 0
        else:
            delay = 0  # Don't apply large delays

        alignment_info = {
            'sign_flip': sign_flip,
            'delay': delay,
            'correlation_after_alignment': np.corrcoef(true_filter, aligned_filter)[0, 1]
        }

        return aligned_filter, alignment_info

    def _compute_cv(self, spike_times: np.ndarray) -> float:
        """Compute coefficient of variation of ISIs."""
        if len(spike_times) < 2:
            return np.nan

        isis = np.diff(spike_times)
        if len(isis) == 0:
            return np.nan

        return np.std(isis) / np.mean(isis)


class ParameterSweep:
    """
    Systematic validation across parameter ranges.
    """

    def __init__(self, random_state: Optional[int] = None):
        """Initialize parameter sweep validator."""
        self.random_state = random_state

    def sweep_noise_levels(self,
                          true_filter: np.ndarray,
                          noise_levels: List[float],
                          n_repeats: int = 5,
                          stimulus_length: int = 10000) -> Dict[str, Any]:
        """
        Test recovery across different noise levels.

        Parameters
        ----------
        true_filter : numpy.ndarray
            Ground truth filter
        noise_levels : list
            List of noise levels to test
        n_repeats : int
            Number of repeats per noise level
        stimulus_length : int
            Length of stimulus

        Returns
        -------
        dict
            Sweep results
        """
        results = {
            'noise_levels': noise_levels,
            'correlations': [],
            'rmse_values': [],
            'snr_values': []
        }

        recovery_validator = GroundTruthRecovery(self.random_state)

        for noise_level in noise_levels:
            noise_correlations = []
            noise_rmse = []
            noise_snr = []

            for repeat in range(n_repeats):
                # Set different seed for each repeat
                if self.random_state is not None:
                    np.random.seed(self.random_state + repeat + int(noise_level * 1000))

                try:
                    recovery_result = recovery_validator.test_filter_recovery(
                        true_filter, stimulus_length, noise_level)

                    if 'whitened_STA' in recovery_result['recovery_metrics']:
                        metrics = recovery_result['recovery_metrics']['whitened_STA']
                        noise_correlations.append(metrics['correlation'])
                        noise_rmse.append(metrics['rmse'])
                        noise_snr.append(metrics['snr'])

                except Exception as e:
                    logger.warning(f"Failed repeat {repeat} at noise {noise_level}: {str(e)}")
                    continue

            # Store mean results for this noise level
            if noise_correlations:
                results['correlations'].append({
                    'mean': np.mean(noise_correlations),
                    'std': np.std(noise_correlations),
                    'values': noise_correlations
                })
                results['rmse_values'].append({
                    'mean': np.mean(noise_rmse),
                    'std': np.std(noise_rmse),
                    'values': noise_rmse
                })
                results['snr_values'].append({
                    'mean': np.mean(noise_snr),
                    'std': np.std(noise_snr),
                    'values': noise_snr
                })
            else:
                results['correlations'].append({'mean': np.nan, 'std': np.nan, 'values': []})
                results['rmse_values'].append({'mean': np.nan, 'std': np.nan, 'values': []})
                results['snr_values'].append({'mean': np.nan, 'std': np.nan, 'values': []})

        return results

    def sweep_data_lengths(self,
                          true_filter: np.ndarray,
                          data_lengths: List[int],
                          n_repeats: int = 3,
                          noise_level: float = 0.1) -> Dict[str, Any]:
        """
        Test recovery across different data lengths.

        Parameters
        ----------
        true_filter : numpy.ndarray
            Ground truth filter
        data_lengths : list
            List of stimulus lengths to test
        n_repeats : int
            Number of repeats per length
        noise_level : float
            Fixed noise level

        Returns
        -------
        dict
            Sweep results
        """
        results = {
            'data_lengths': data_lengths,
            'correlations': [],
            'spike_counts': []
        }

        recovery_validator = GroundTruthRecovery(self.random_state)

        for length in data_lengths:
            length_correlations = []
            length_spike_counts = []

            for repeat in range(n_repeats):
                if self.random_state is not None:
                    np.random.seed(self.random_state + repeat + length)

                try:
                    recovery_result = recovery_validator.test_filter_recovery(
                        true_filter, length, noise_level)

                    if 'whitened_STA' in recovery_result['recovery_metrics']:
                        corr = recovery_result['recovery_metrics']['whitened_STA']['correlation']
                        length_correlations.append(corr)
                        length_spike_counts.append(recovery_result['spike_stats']['count'])

                except Exception as e:
                    logger.warning(f"Failed repeat {repeat} at length {length}: {str(e)}")
                    continue

            if length_correlations:
                results['correlations'].append({
                    'mean': np.mean(length_correlations),
                    'std': np.std(length_correlations),
                    'values': length_correlations
                })
                results['spike_counts'].append({
                    'mean': np.mean(length_spike_counts),
                    'std': np.std(length_spike_counts),
                    'values': length_spike_counts
                })
            else:
                results['correlations'].append({'mean': np.nan, 'std': np.nan, 'values': []})
                results['spike_counts'].append({'mean': np.nan, 'std': np.nan, 'values': []})

        return results


def create_validation_report(validation_results: Dict[str, Any],
                           output_path: Optional[str] = None) -> str:
    """
    Create a formatted validation report.

    Parameters
    ----------
    validation_results : dict
        Results from validation tests
    output_path : str, optional
        Path to save report

    Returns
    -------
    str
        Formatted report text
    """
    report_lines = []
    report_lines.append("White Noise Analysis Validation Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Ground truth recovery
    if 'ground_truth_recovery' in validation_results:
        gt_results = validation_results['ground_truth_recovery']
        report_lines.append("Ground Truth Recovery:")

        if 'recovery_metrics' in gt_results:
            for method, metrics in gt_results['recovery_metrics'].items():
                report_lines.append(f"  {method}:")
                report_lines.append(f"    Correlation: {metrics['correlation']:.3f}")
                report_lines.append(f"    RMSE: {metrics['rmse']:.3f}")
                report_lines.append(f"    SNR: {metrics['snr']:.1f} dB")

        report_lines.append("")

    # Parameter sweeps
    if 'noise_sweep' in validation_results:
        noise_results = validation_results['noise_sweep']
        report_lines.append("Noise Level Sweep:")

        for i, noise_level in enumerate(noise_results['noise_levels']):
            if i < len(noise_results['correlations']):
                corr_data = noise_results['correlations'][i]
                report_lines.append(f"  Noise {noise_level:.3f}: "
                                  f"Correlation {corr_data['mean']:.3f} ± {corr_data['std']:.3f}")

        report_lines.append("")

    if 'length_sweep' in validation_results:
        length_results = validation_results['length_sweep']
        report_lines.append("Data Length Sweep:")

        for i, length in enumerate(length_results['data_lengths']):
            if i < len(length_results['correlations']):
                corr_data = length_results['correlations'][i]
                spike_data = length_results['spike_counts'][i]
                report_lines.append(f"  Length {length}: "
                                  f"Correlation {corr_data['mean']:.3f}, "
                                  f"Spikes {spike_data['mean']:.0f}")

        report_lines.append("")

    report_text = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Validation report saved to {output_path}")

    return report_text


def run_comprehensive_validation(filter_length: int = 20,
                               stimulus_length: int = 15000,
                               random_state: Optional[int] = 42) -> Dict[str, Any]:
    """
    Run a comprehensive validation suite.

    Parameters
    ----------
    filter_length : int
        Length of test filter
    stimulus_length : int
        Length of stimulus
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Comprehensive validation results
    """
    logger.info("Starting comprehensive validation")

    # Create a realistic test filter
    t = np.arange(filter_length) * 0.001
    true_filter = np.exp(-t/0.008) * np.sin(2*np.pi*t/0.004)
    true_filter = true_filter / np.linalg.norm(true_filter)

    results = {}

    # Ground truth recovery test
    gt_validator = GroundTruthRecovery(random_state)
    try:
        results['ground_truth_recovery'] = gt_validator.test_filter_recovery(
            true_filter, stimulus_length)
        logger.info("Ground truth recovery test completed")
    except Exception as e:
        logger.error(f"Ground truth recovery test failed: {str(e)}")
        results['ground_truth_recovery'] = {'error': str(e)}

    # Noise level sweep
    param_sweeper = ParameterSweep(random_state)
    try:
        noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
        results['noise_sweep'] = param_sweeper.sweep_noise_levels(
            true_filter, noise_levels, n_repeats=3, stimulus_length=stimulus_length//2)
        logger.info("Noise level sweep completed")
    except Exception as e:
        logger.error(f"Noise level sweep failed: {str(e)}")
        results['noise_sweep'] = {'error': str(e)}

    # Data length sweep
    try:
        data_lengths = [5000, 10000, 20000, 30000]
        results['length_sweep'] = param_sweeper.sweep_data_lengths(
            true_filter, data_lengths, n_repeats=2)
        logger.info("Data length sweep completed")
    except Exception as e:
        logger.error(f"Data length sweep failed: {str(e)}")
        results['length_sweep'] = {'error': str(e)}

    # Nonlinearity recovery test
    try:
        nonlinearity_params = {
            'type': 'exponential',
            'threshold': 0.0,
            'slope': 1.0
        }
        results['nonlinearity_recovery'] = gt_validator.test_nonlinearity_recovery(
            nonlinearity_params, filter_length, stimulus_length)
        logger.info("Nonlinearity recovery test completed")
    except Exception as e:
        logger.error(f"Nonlinearity recovery test failed: {str(e)}")
        results['nonlinearity_recovery'] = {'error': str(e)}

    logger.info("Comprehensive validation completed")
    return results

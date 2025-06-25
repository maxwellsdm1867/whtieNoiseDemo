"""
Metrics and evaluation utilities for white noise analysis.

This module provides metrics for evaluating the quality of filter estimates,
nonlinearity fits, and model predictions.
"""

import warnings
from typing import Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
from scipy import stats, optimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..core.exceptions import DataValidationError, ProcessingError
from .logging_config import get_logger

logger = get_logger(__name__)


class FilterMetrics:
    """
    Metrics for evaluating linear filter quality.
    
    Can be used as a static class or instantiated with a filter for convenience.
    """
    
    def __init__(self, filter_estimate: Optional[np.ndarray] = None):
        """
        Initialize FilterMetrics with optional filter.
        
        Parameters
        ----------
        filter_estimate : numpy.ndarray, optional
            Filter to analyze. If provided, instance methods can be used.
        """
        self.filter = filter_estimate
    
    def compute_snr(self, 
                   noise_estimate: Optional[np.ndarray] = None,
                   noise_floor: Optional[float] = None) -> float:
        """
        Compute signal-to-noise ratio of stored filter.
        
        Parameters
        ----------
        noise_estimate : numpy.ndarray, optional
            Noise estimate
        noise_floor : float, optional
            Constant noise floor estimate
            
        Returns
        -------
        float
            Signal-to-noise ratio in dB
        """
        if self.filter is None:
            raise ValueError("No filter provided. Use FilterMetrics(filter) or static methods.")
        return self.signal_to_noise_ratio(self.filter, noise_estimate, noise_floor)
    
    def compute_peak_properties(self) -> Dict[str, Any]:
        """
        Compute peak properties of stored filter.
        
        Returns
        -------
        dict
            Dictionary with peak_amplitude, peak_time, peak_width
        """
        if self.filter is None:
            raise ValueError("No filter provided. Use FilterMetrics(filter) or static methods.")
            
        abs_filter = np.abs(self.filter)
        peak_amplitude = float(np.max(abs_filter))
        peak_time = int(np.argmax(abs_filter))
        
        # Compute peak width (full width at half maximum)
        half_max = peak_amplitude / 2
        above_half = abs_filter >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            peak_width = indices[-1] - indices[0] + 1
        else:
            peak_width = 1
            
        return {
            'peak_amplitude': peak_amplitude,
            'peak_time': peak_time,
            'peak_width': int(peak_width)
        }
    
    def compute_quality_score(self) -> float:
        """
        Compute overall quality score for stored filter.
        
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        if self.filter is None:
            raise ValueError("No filter provided. Use FilterMetrics(filter) or static methods.")
            
        # Compute SNR component
        snr = self.compute_snr()
        snr_score = np.tanh(snr / 10.0) if np.isfinite(snr) and snr > 0 else 0.0
        
        # Compute peak properties component
        peak_props = self.compute_peak_properties()
        peak_score = min(1.0, peak_props['peak_amplitude'] / 1.0)  # Normalize to reasonable range
        
        # Combined score
        quality = 0.6 * snr_score + 0.4 * peak_score
        return float(np.clip(quality, 0.0, 1.0))

    @staticmethod
    def signal_to_noise_ratio(filter_estimate: np.ndarray,
                            noise_estimate: Optional[np.ndarray] = None,
                            noise_floor: Optional[float] = None) -> float:
        """
        Compute signal-to-noise ratio of filter estimate.

        Parameters
        ----------
        filter_estimate : numpy.ndarray
            Estimated filter
        noise_estimate : numpy.ndarray, optional
            Noise estimate (e.g., from surrogate data)
        noise_floor : float, optional
            Constant noise floor estimate

        Returns
        -------
        float
            Signal-to-noise ratio in dB
        """
        signal_power = np.var(filter_estimate)

        if noise_estimate is not None:
            noise_power = np.var(noise_estimate)
        elif noise_floor is not None:
            noise_power = noise_floor ** 2
        else:
            # Estimate noise from high frequencies
            if len(filter_estimate) > 10:
                # Assume last 20% is mostly noise
                noise_start = int(0.8 * len(filter_estimate))
                noise_power = np.var(filter_estimate[noise_start:])
            else:
                noise_power = signal_power * 0.1  # Assume 10% noise

        if noise_power <= 0:
            return np.inf

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    @staticmethod
    def filter_consistency(filter1: np.ndarray,
                          filter2: np.ndarray,
                          metric: str = 'correlation') -> float:
        """
        Measure consistency between two filter estimates.

        Parameters
        ----------
        filter1, filter2 : numpy.ndarray
            Filter estimates to compare
        metric : str
            Consistency metric: 'correlation', 'cosine', 'rmse'

        Returns
        -------
        float
            Consistency measure
        """
        if len(filter1) != len(filter2):
            min_len = min(len(filter1), len(filter2))
            filter1 = filter1[:min_len]
            filter2 = filter2[:min_len]

        if metric == 'correlation':
            try:
                # Use numpy correlation which is more straightforward
                corr = np.corrcoef(filter1, filter2)[0, 1]
                if np.isfinite(corr):
                    return float(corr)
                else:
                    return 0.0
            except Exception:
                return 0.0

        elif metric == 'cosine':
            norm1 = np.linalg.norm(filter1)
            norm2 = np.linalg.norm(filter2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(filter1, filter2) / (norm1 * norm2))

        elif metric == 'rmse':
            return float(np.sqrt(np.mean((filter1 - filter2) ** 2)))

        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def compute_latency(filter_signal: np.ndarray,
                       sampling_rate: float,
                       threshold: float = 0.5) -> Tuple[float, float]:
        """
        Compute response latency from filter.

        Parameters
        ----------
        filter_signal : numpy.ndarray
            Filter estimate
        sampling_rate : float
            Sampling rate (Hz)
        threshold : float
            Threshold for peak detection (fraction of max)

        Returns
        -------
        peak_latency : float
            Time to peak response (ms)
        onset_latency : float
            Time to response onset (ms)
        """
        abs_filter = np.abs(filter_signal)
        max_val = np.max(abs_filter)

        if max_val == 0:
            return np.nan, np.nan

        # Peak latency
        peak_idx = np.argmax(abs_filter)
        peak_latency = peak_idx / sampling_rate * 1000  # ms

        # Onset latency (first point above threshold)
        threshold_val = threshold * max_val
        onset_indices = np.where(abs_filter >= threshold_val)[0]

        if len(onset_indices) > 0:
            onset_latency = onset_indices[0] / sampling_rate * 1000  # ms
        else:
            onset_latency = np.nan

        return float(peak_latency), float(onset_latency)

    @staticmethod
    def temporal_profile_metrics(filter_signal: np.ndarray,
                               sampling_rate: float) -> Dict[str, float]:
        """
        Compute temporal profile metrics of filter.

        Parameters
        ----------
        filter_signal : numpy.ndarray
            Filter estimate
        sampling_rate : float
            Sampling rate (Hz)

        Returns
        -------
        dict
            Dictionary of temporal metrics
        """
        if len(filter_signal) == 0:
            return {'duration': 0.0, 'peak_latency': np.nan, 'onset_latency': np.nan,
                   'biphasic_index': 0.0, 'temporal_width': 0.0}

        peak_latency, onset_latency = FilterMetrics.compute_latency(
            filter_signal, sampling_rate)

        # Duration (full width)
        duration = len(filter_signal) / sampling_rate * 1000  # ms

        # Biphasic index (measure of positive vs negative components)
        pos_sum = np.sum(filter_signal[filter_signal > 0])
        neg_sum = np.abs(np.sum(filter_signal[filter_signal < 0]))
        total_sum = pos_sum + neg_sum

        if total_sum > 0:
            biphasic_index = min(pos_sum, neg_sum) / total_sum
        else:
            biphasic_index = 0.0

        # Temporal width (width at half maximum)
        abs_filter = np.abs(filter_signal)
        max_val = np.max(abs_filter)

        if max_val > 0:
            half_max = max_val / 2
            above_half = abs_filter >= half_max
            if np.any(above_half):
                first_idx = np.where(above_half)[0][0]
                last_idx = np.where(above_half)[0][-1]
                temporal_width = (last_idx - first_idx) / sampling_rate * 1000  # ms
            else:
                temporal_width = 0.0
        else:
            temporal_width = 0.0

        return {
            'duration': float(duration),
            'peak_latency': peak_latency,
            'onset_latency': onset_latency,
            'biphasic_index': float(biphasic_index),
            'temporal_width': float(temporal_width)
        }


class NonlinearityMetrics:
    """
    Metrics for evaluating nonlinearity estimates.
    
    Can be used as a static class or instantiated with x,y data for convenience.
    """
    
    def __init__(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """
        Initialize NonlinearityMetrics with optional data.
        
        Parameters
        ----------
        x : numpy.ndarray, optional
            Input values (generator signal)
        y : numpy.ndarray, optional
            Output values (firing rate)
        """
        self.x = x
        self.y = y
    
    def compute_dynamic_range(self) -> float:
        """
        Compute dynamic range of stored nonlinearity.
        
        Returns
        -------
        float
            Dynamic range in log units
        """
        if self.y is None:
            raise ValueError("No y data provided. Use NonlinearityMetrics(x, y) or static methods.")
            
        y_clean = self.y[np.isfinite(self.y)]
        if len(y_clean) < 2:
            return np.nan
            
        y_positive = y_clean[y_clean > 0]
        if len(y_positive) == 0:
            return 0.0
            
        max_val = np.max(y_positive)
        min_val = np.min(y_positive)
        
        if min_val <= 0:
            min_val = np.min(y_positive[y_positive > 0]) if np.any(y_positive > 0) else max_val * 1e-6
            
        dynamic_range = np.log(max_val / min_val)
        return float(dynamic_range)
    
    def compute_threshold(self) -> float:
        """
        Compute threshold of stored nonlinearity.
        
        Returns
        -------
        float
            Threshold value where response begins
        """
        if self.x is None or self.y is None:
            raise ValueError("No x,y data provided. Use NonlinearityMetrics(x, y) or static methods.")
            
        # Find where response starts to rise significantly above baseline
        threshold, _ = self.threshold_detection(self.x, self.y)
        return threshold
    
    def compute_saturation_level(self) -> float:
        """
        Compute saturation level of stored nonlinearity.
        
        Returns
        -------
        float
            Saturation level (95th percentile of response)
        """
        if self.y is None:
            raise ValueError("No y data provided. Use NonlinearityMetrics(x, y) or static methods.")
            
        y_clean = self.y[np.isfinite(self.y)]
        if len(y_clean) == 0:
            return np.nan
            
        saturation_level = np.percentile(y_clean, 95)
        return float(saturation_level)
    
    def analyze_shape(self) -> Dict[str, Any]:
        """
        Analyze shape properties of stored nonlinearity.
        
        Returns
        -------
        dict
            Dictionary with monotonic, convex, symmetric properties
        """
        if self.x is None or self.y is None:
            raise ValueError("No x,y data provided. Use NonlinearityMetrics(x, y) or static methods.")
            
        # Sort by x for analysis
        sort_indices = np.argsort(self.x)
        x_sorted = self.x[sort_indices]
        y_sorted = self.y[sort_indices]
        
        # Remove invalid points
        valid_mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
        x_clean = x_sorted[valid_mask]
        y_clean = y_sorted[valid_mask]
        
        if len(x_clean) < 3:
            return {'monotonic': False, 'convex': False, 'symmetric': False}
        
        # Check monotonicity
        dy = np.diff(y_clean)
        monotonic = np.all(dy >= 0) or np.all(dy <= 0)
        
        # Check convexity (second derivative test)
        if len(y_clean) >= 3:
            d2y = np.diff(y_clean, n=2)
            convex = np.all(d2y >= 0) or np.all(d2y <= 0)
        else:
            convex = False
            
        # Check symmetry around zero (approximate)
        if len(x_clean) > 5:
            # Find center
            x_center = np.median(x_clean)
            x_centered = x_clean - x_center
            
            # Check if y values are similar for +/- x values
            pos_indices = x_centered > 0
            neg_indices = x_centered < 0
            
            if np.any(pos_indices) and np.any(neg_indices):
                # Rough symmetry check
                pos_mean = np.mean(y_clean[pos_indices])
                neg_mean = np.mean(y_clean[neg_indices])
                symmetric = abs(pos_mean - neg_mean) < 0.1 * (abs(pos_mean) + abs(neg_mean))
            else:
                symmetric = False
        else:
            symmetric = False
            
        return {
            'monotonic': bool(monotonic),
            'convex': bool(convex),
            'symmetric': bool(symmetric)
        }

    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R-squared for nonlinearity fit.

        Parameters
        ----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values

        Returns
        -------
        float
            R-squared value
        """
        if len(y_true) != len(y_pred):
            raise DataValidationError("Arrays must have same length")

        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(valid_mask) < 2:
            return np.nan

        return float(r2_score(y_true[valid_mask], y_pred[valid_mask]))

    @staticmethod
    def nonlinearity_strength(x_values: np.ndarray,
                            y_values: np.ndarray) -> float:
        """
        Measure strength of nonlinearity.

        Parameters
        ----------
        x_values : numpy.ndarray
            Input values (generator signal)
        y_values : numpy.ndarray
            Output values (firing rate)

        Returns
        -------
        float
            Nonlinearity strength (0 = linear, 1 = highly nonlinear)
        """
        if len(x_values) != len(y_values):
            raise DataValidationError("Arrays must have same length")

        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if np.sum(valid_mask) < 3:
            return np.nan

        x_clean = x_values[valid_mask]
        y_clean = y_values[valid_mask]

        # Fit linear model
        linear_slope, linear_intercept, linear_r, _, _ = stats.linregress(x_clean, y_clean)
        linear_pred = linear_slope * x_clean + linear_intercept

        # Compute deviations from linearity
        residuals = y_clean - linear_pred
        linear_variance = np.var(linear_pred)
        residual_variance = np.var(residuals)

        if linear_variance + residual_variance == 0:
            return 0.0

        nonlinearity_strength = residual_variance / (linear_variance + residual_variance)
        return float(nonlinearity_strength)

    @staticmethod
    def threshold_detection(x_values: np.ndarray,
                          y_values: np.ndarray,
                          method: str = 'derivative') -> Tuple[float, float]:
        """
        Detect threshold and saturation points.

        Parameters
        ----------
        x_values : numpy.ndarray
            Input values
        y_values : numpy.ndarray
            Output values
        method : str
            Detection method: 'derivative', 'inflection'

        Returns
        -------
        threshold : float
            Threshold point
        saturation : float
            Saturation point
        """
        if len(x_values) != len(y_values):
            raise DataValidationError("Arrays must have same length")

        # Sort by x values
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        y_sorted = y_values[sort_idx]

        if method == 'derivative':
            # Find points of maximum derivative
            if len(x_sorted) < 3:
                return np.nan, np.nan

            dy_dx = np.gradient(y_sorted, x_sorted)

            # Threshold: first significant increase
            max_deriv = np.max(dy_dx)
            threshold_idx = np.argmax(dy_dx > 0.1 * max_deriv)
            threshold = float(x_sorted[threshold_idx])

            # Saturation: where derivative drops significantly
            sat_indices = np.where(dy_dx < 0.1 * max_deriv)[0]
            sat_indices = sat_indices[sat_indices > threshold_idx]

            if len(sat_indices) > 0:
                saturation = float(x_sorted[sat_indices[0]])
            else:
                saturation = float(np.max(x_sorted))

        elif method == 'inflection':
            # Find inflection points using second derivative
            if len(x_sorted) < 5:
                return np.nan, np.nan

            d2y_dx2 = np.gradient(np.gradient(y_sorted, x_sorted), x_sorted)

            # Find zero crossings of second derivative
            zero_crossings = np.where(np.diff(np.signbit(d2y_dx2)))[0]

            if len(zero_crossings) >= 2:
                threshold = float(x_sorted[zero_crossings[0]])
                saturation = float(x_sorted[zero_crossings[-1]])
            elif len(zero_crossings) == 1:
                threshold = float(x_sorted[zero_crossings[0]])
                saturation = float(np.max(x_sorted))
            else:
                threshold = float(np.min(x_sorted))
                saturation = float(np.max(x_sorted))

        else:
            raise ValueError(f"Unknown method: {method}")

        return threshold, saturation


class ModelMetrics:
    """
    Metrics for evaluating full LN model performance.
    """

    @staticmethod
    def prediction_accuracy(spike_times: np.ndarray,
                          predicted_rate: np.ndarray,
                          time_bins: np.ndarray,
                          metric: str = 'correlation') -> float:
        """
        Compute prediction accuracy of LN model.

        Parameters
        ----------
        spike_times : numpy.ndarray
            Observed spike times
        predicted_rate : numpy.ndarray
            Predicted firing rate
        time_bins : numpy.ndarray
            Time bins for rate computation
        metric : str
            Accuracy metric: 'correlation', 'rmse', 'poisson_likelihood'

        Returns
        -------
        float
            Prediction accuracy
        """
        # Compute observed rate
        dt = np.mean(np.diff(time_bins))
        observed_rate = np.zeros(len(time_bins))

        for i, t in enumerate(time_bins):
            spike_count = np.sum((spike_times >= t - dt/2) & (spike_times < t + dt/2))
            observed_rate[i] = spike_count / dt

        valid_mask = np.isfinite(predicted_rate) & np.isfinite(observed_rate)
        if np.sum(valid_mask) < 2:
            return np.nan

        obs_clean = observed_rate[valid_mask]
        pred_clean = predicted_rate[valid_mask]

        if metric == 'correlation':
            if np.var(obs_clean) == 0 or np.var(pred_clean) == 0:
                return np.nan
            try:
                corr = np.corrcoef(obs_clean, pred_clean)[0, 1]
                if np.isfinite(corr):
                    return float(corr)
                else:
                    return 0.0
            except Exception:
                return 0.0

        elif metric == 'rmse':
            return float(np.sqrt(np.mean((obs_clean - pred_clean) ** 2)))

        elif metric == 'poisson_likelihood':
            # Avoid numerical issues
            pred_clean = np.maximum(pred_clean, 1e-10)
            log_likelihood = np.sum(obs_clean * np.log(pred_clean * dt) - pred_clean * dt)
            return float(log_likelihood)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def cross_validate_model(stimulus: np.ndarray,
                           spike_times: np.ndarray,
                           model_func: Callable,
                           n_folds: int = 5,
                           random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Cross-validate model performance.

        Parameters
        ----------
        stimulus : numpy.ndarray
            Stimulus data
        spike_times : numpy.ndarray
            Spike times
        model_func : callable
            Function that takes (train_stim, train_spikes) and returns model
        n_folds : int
            Number of cross-validation folds
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Cross-validation results
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Create time-based folds to preserve temporal structure
        duration = len(stimulus)
        fold_size = duration // n_folds

        correlations = []
        rmse_values = []

        for fold in range(n_folds):
            # Define test indices
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, duration)

            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, duration)
            ])
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            # Split data
            train_stimulus = stimulus[train_indices]
            test_stimulus = stimulus[test_indices]

            # Convert test_indices to time
            test_times = test_indices  # Assuming 1 sample per time unit
            train_spikes = spike_times[
                (spike_times >= 0) & (spike_times < train_indices[-1])]
            test_spikes = spike_times[
                (spike_times >= test_start) & (spike_times < test_end)] - test_start

            try:
                # Train model
                model = model_func(train_stimulus, train_spikes)

                # Predict on test set
                predicted_rate = model.predict(test_stimulus)

                # Compute metrics
                time_bins = np.arange(len(test_stimulus))
                corr = ModelMetrics.prediction_accuracy(
                    test_spikes, predicted_rate, time_bins, 'correlation')
                rmse = ModelMetrics.prediction_accuracy(
                    test_spikes, predicted_rate, time_bins, 'rmse')

                if not np.isnan(corr):
                    correlations.append(corr)
                if not np.isnan(rmse):
                    rmse_values.append(rmse)

            except Exception as e:
                logger.warning(f"Fold {fold} failed: {str(e)}")
                continue

        results = {
            'mean_correlation': float(np.mean(correlations)) if correlations else np.nan,
            'std_correlation': float(np.std(correlations)) if correlations else np.nan,
            'mean_rmse': float(np.mean(rmse_values)) if rmse_values else np.nan,
            'std_rmse': float(np.std(rmse_values)) if rmse_values else np.nan,
            'n_successful_folds': len(correlations)
        }

        return results


def compute_summary_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive summary statistics.

    Parameters
    ----------
    data : numpy.ndarray
        Input data

    Returns
    -------
    dict
        Summary statistics
    """
    if len(data) == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'median': np.nan, 'q25': np.nan, 'q75': np.nan,
            'skewness': np.nan, 'kurtosis': np.nan
        }

    valid_data = data[np.isfinite(data)]

    if len(valid_data) == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'median': np.nan, 'q25': np.nan, 'q75': np.nan,
            'skewness': np.nan, 'kurtosis': np.nan
        }

    stats_dict = {
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'median': float(np.median(valid_data)),
        'q25': float(np.percentile(valid_data, 25)),
        'q75': float(np.percentile(valid_data, 75))
    }

    # Handle edge cases for skewness and kurtosis
    if len(valid_data) >= 3 and np.var(valid_data) > 0:
        stats_dict['skewness'] = float(stats.skew(valid_data))
    else:
        stats_dict['skewness'] = np.nan

    if len(valid_data) >= 4 and np.var(valid_data) > 0:
        stats_dict['kurtosis'] = float(stats.kurtosis(valid_data))
    else:
        stats_dict['kurtosis'] = np.nan

    return stats_dict


def bootstrap_confidence_interval(data: np.ndarray,
                                statistic_func: Callable,
                                n_bootstrap: int = 1000,
                                confidence: float = 0.95,
                                random_state: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : numpy.ndarray
        Input data
    statistic_func : callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (0-1)
    random_state : int, optional
        Random seed

    Returns
    -------
    statistic : float
        Original statistic value
    lower_ci : float
        Lower confidence interval
    upper_ci : float
        Upper confidence interval
    """
    if random_state is not None:
        np.random.seed(random_state)

    original_stat = statistic_func(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(data), size=len(data), replace=True)
        bootstrap_data = data[indices]

        try:
            stat = statistic_func(bootstrap_data)
            if np.isfinite(stat):
                bootstrap_stats.append(stat)
        except Exception:
            continue

    if len(bootstrap_stats) == 0:
        return original_stat, np.nan, np.nan

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_ci = np.percentile(bootstrap_stats, lower_percentile)
    upper_ci = np.percentile(bootstrap_stats, upper_percentile)

    return float(original_stat), float(lower_ci), float(upper_ci)

"""
Preprocessing utilities for neuronal data.

This module provides utilities for cleaning, filtering, and preprocessing
neuronal data before white noise analysis.
"""

import warnings
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from scipy import signal, stats
from scipy.ndimage import median_filter
import logging

from ..core.exceptions import DataValidationError, ProcessingError
from .logging_config import get_logger

logger = get_logger(__name__)


class SpikeProcessor:
    """
    Utilities for spike data preprocessing.
    """
    
    @staticmethod
    def remove_artifacts(spike_times: np.ndarray,
                        artifact_window: float = 0.001,
                        min_isi: float = 0.0005) -> Tuple[np.ndarray, int]:
        """
        Remove artifacts from spike times.
        
        Parameters
        ----------
        spike_times : numpy.ndarray
            Array of spike times in seconds
        artifact_window : float
            Window around artifacts to remove (seconds)
        min_isi : float
            Minimum inter-spike interval to consider valid (seconds)
            
        Returns
        -------
        cleaned_spikes : numpy.ndarray
            Cleaned spike times
        removed_count : int
            Number of spikes removed
        """
        if len(spike_times) == 0:
            return spike_times, 0
        
        original_count = len(spike_times)
        spike_times = np.sort(spike_times)
        
        # Remove spikes with very short ISIs
        isis = np.diff(np.concatenate([[0], spike_times, [np.inf]]))
        valid_mask = (isis[:-1] >= min_isi) & (isis[1:] >= min_isi)
        
        cleaned_spikes = spike_times[valid_mask]
        removed_count = original_count - len(cleaned_spikes)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} artifact spikes")
        
        return cleaned_spikes, removed_count
    
    @staticmethod
    def detect_bursts(spike_times: np.ndarray,
                     max_isi: float = 0.01,
                     min_burst_length: int = 3) -> Dict[str, np.ndarray]:
        """
        Detect burst activity in spike trains.
        
        Parameters
        ----------
        spike_times : numpy.ndarray
            Array of spike times
        max_isi : float
            Maximum ISI within a burst (seconds)
        min_burst_length : int
            Minimum number of spikes in a burst
            
        Returns
        -------
        dict
            Dictionary with burst information:
            - 'burst_starts': Start times of bursts
            - 'burst_ends': End times of bursts
            - 'burst_spikes': Spike indices in bursts
            - 'isolated_spikes': Spike indices not in bursts
        """
        if len(spike_times) < min_burst_length:
            return {
                'burst_starts': np.array([]),
                'burst_ends': np.array([]),
                'burst_spikes': np.array([], dtype=int),
                'isolated_spikes': np.arange(len(spike_times))
            }
        
        isis = np.diff(spike_times)
        in_burst = isis <= max_isi
        
        # Find burst boundaries
        burst_changes = np.diff(np.concatenate([[False], in_burst, [False]]).astype(int))
        burst_starts_idx = np.where(burst_changes == 1)[0]
        burst_ends_idx = np.where(burst_changes == -1)[0]
        
        # Filter by minimum burst length
        valid_bursts = []
        burst_spike_indices = []
        
        for start_idx, end_idx in zip(burst_starts_idx, burst_ends_idx):
            burst_length = end_idx - start_idx + 1
            if burst_length >= min_burst_length:
                valid_bursts.append((start_idx, end_idx))
                burst_spike_indices.extend(range(start_idx, end_idx + 1))
        
        if valid_bursts:
            burst_starts = np.array([spike_times[start] for start, _ in valid_bursts])
            burst_ends = np.array([spike_times[end] for _, end in valid_bursts])
            burst_spikes = np.array(burst_spike_indices)
        else:
            burst_starts = np.array([])
            burst_ends = np.array([])
            burst_spikes = np.array([], dtype=int)
        
        isolated_spikes = np.setdiff1d(np.arange(len(spike_times)), burst_spikes)
        
        return {
            'burst_starts': burst_starts,
            'burst_ends': burst_ends,
            'burst_spikes': burst_spikes,
            'isolated_spikes': isolated_spikes
        }
    
    @staticmethod
    def compute_firing_rate(spike_times: np.ndarray,
                          duration: float,
                          window_size: float = 1.0,
                          step_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous firing rate.
        
        Parameters
        ----------
        spike_times : numpy.ndarray
            Spike times
        duration : float
            Total duration of recording
        window_size : float
            Size of sliding window (seconds)
        step_size : float, optional
            Step size for sliding window. If None, uses window_size/4
            
        Returns
        -------
        time_bins : numpy.ndarray
            Time points for firing rate
        firing_rate : numpy.ndarray
            Instantaneous firing rate (Hz)
        """
        if step_size is None:
            step_size = window_size / 4
        
        time_bins = np.arange(window_size/2, duration - window_size/2, step_size)
        firing_rate = np.zeros(len(time_bins))
        
        for i, t in enumerate(time_bins):
            start_time = t - window_size/2
            end_time = t + window_size/2
            
            spike_count = np.sum((spike_times >= start_time) & (spike_times < end_time))
            firing_rate[i] = spike_count / window_size
        
        return time_bins, firing_rate


class StimulusProcessor:
    """
    Utilities for stimulus preprocessing.
    """
    
    @staticmethod
    def normalize_stimulus(stimulus: np.ndarray,
                          method: str = 'zscore',
                          axis: Optional[int] = None) -> np.ndarray:
        """
        Normalize stimulus data.
        
        Parameters
        ----------
        stimulus : numpy.ndarray
            Stimulus data
        method : str
            Normalization method: 'zscore', 'minmax', 'robust'
        axis : int, optional
            Axis along which to normalize
            
        Returns
        -------
        numpy.ndarray
            Normalized stimulus
        """
        if method == 'zscore':
            if axis is None:
                return stats.zscore(stimulus, nan_policy='omit')
            else:
                return stats.zscore(stimulus, axis=axis, nan_policy='omit')
        
        elif method == 'minmax':
            if axis is None:
                min_val = np.nanmin(stimulus)
                max_val = np.nanmax(stimulus)
            else:
                min_val = np.nanmin(stimulus, axis=axis, keepdims=True)
                max_val = np.nanmax(stimulus, axis=axis, keepdims=True)
            
            return (stimulus - min_val) / (max_val - min_val + 1e-12)
        
        elif method == 'robust':
            if axis is None:
                median = np.nanmedian(stimulus)
                mad = np.nanmedian(np.abs(stimulus - median))
            else:
                median = np.nanmedian(stimulus, axis=axis, keepdims=True)
                mad = np.nanmedian(np.abs(stimulus - median), axis=axis, keepdims=True)
            
            return (stimulus - median) / (mad + 1e-12)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def filter_stimulus(stimulus: np.ndarray,
                       sampling_rate: float,
                       lowpass: Optional[float] = None,
                       highpass: Optional[float] = None,
                       filter_type: str = 'butterworth',
                       order: int = 4) -> np.ndarray:
        """
        Apply frequency filtering to stimulus.
        
        Parameters
        ----------
        stimulus : numpy.ndarray
            Input stimulus
        sampling_rate : float
            Sampling rate (Hz)
        lowpass : float, optional
            Lowpass cutoff frequency (Hz)
        highpass : float, optional
            Highpass cutoff frequency (Hz)
        filter_type : str
            Filter type: 'butterworth', 'chebyshev', 'elliptic'
        order : int
            Filter order
            
        Returns
        -------
        numpy.ndarray
            Filtered stimulus
        """
        nyquist = sampling_rate / 2
        
        if lowpass is not None and lowpass >= nyquist:
            warnings.warn(f"Lowpass frequency {lowpass} >= Nyquist {nyquist}")
            lowpass = None
        
        if highpass is not None and highpass >= nyquist:
            warnings.warn(f"Highpass frequency {highpass} >= Nyquist {nyquist}")
            highpass = None
        
        if lowpass is None and highpass is None:
            return stimulus
        
        # Design filter and apply
        if lowpass is not None and highpass is not None:
            # Bandpass
            if filter_type == 'butterworth':
                sos = signal.butter(order, [highpass/nyquist, lowpass/nyquist], 
                                  btype='band', output='sos')
            elif filter_type == 'chebyshev':
                sos = signal.cheby1(order, 0.1, [highpass/nyquist, lowpass/nyquist], 
                                  btype='band', output='sos')
            elif filter_type == 'elliptic':
                sos = signal.ellip(order, 0.1, 40, [highpass/nyquist, lowpass/nyquist], 
                                 btype='band', output='sos')
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
        elif lowpass is not None:
            # Lowpass
            if filter_type == 'butterworth':
                sos = signal.butter(order, lowpass/nyquist, btype='low', output='sos')
            elif filter_type == 'chebyshev':
                sos = signal.cheby1(order, 0.1, lowpass/nyquist, btype='low', output='sos')
            elif filter_type == 'elliptic':
                sos = signal.ellip(order, 0.1, 40, lowpass/nyquist, btype='low', output='sos')
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
        else:
            # Highpass - highpass is guaranteed to be not None here
            assert highpass is not None  # Type checker hint
            if filter_type == 'butterworth':
                sos = signal.butter(order, highpass/nyquist, btype='high', output='sos')
            elif filter_type == 'chebyshev':
                sos = signal.cheby1(order, 0.1, highpass/nyquist, btype='high', output='sos')
            elif filter_type == 'elliptic':
                sos = signal.ellip(order, 0.1, 40, highpass/nyquist, btype='high', output='sos')
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
        
        filtered = signal.sosfilt(sos, stimulus, axis=-1)
        # Ensure we return ndarray, not tuple
        if isinstance(filtered, tuple):
            return filtered[0]
        return filtered
    
    @staticmethod
    def remove_outliers(stimulus: np.ndarray,
                       method: str = 'zscore',
                       threshold: float = 3.0,
                       axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from stimulus data.
        
        Parameters
        ----------
        stimulus : numpy.ndarray
            Input stimulus
        method : str
            Outlier detection method: 'zscore', 'iqr', 'mad'
        threshold : float
            Threshold for outlier detection
        axis : int, optional
            Axis along which to detect outliers
            
        Returns
        -------
        cleaned_stimulus : numpy.ndarray
            Stimulus with outliers replaced by NaN
        outlier_mask : numpy.ndarray
            Boolean mask indicating outliers
        """
        if method == 'zscore':
            if axis is None:
                z_scores = np.abs(stats.zscore(stimulus, nan_policy='omit'))
            else:
                z_scores = np.abs(stats.zscore(stimulus, axis=axis, nan_policy='omit'))
            outlier_mask = z_scores > threshold
        
        elif method == 'iqr':
            if axis is None:
                q1, q3 = np.nanpercentile(stimulus, [25, 75])
            else:
                q1, q3 = np.nanpercentile(stimulus, [25, 75], axis=axis, keepdims=True)
            
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (stimulus < lower_bound) | (stimulus > upper_bound)
        
        elif method == 'mad':
            if axis is None:
                median_val = np.nanmedian(stimulus)
                mad = np.nanmedian(np.abs(stimulus - median_val))
            else:
                median_val = np.nanmedian(stimulus, axis=axis, keepdims=True)
                mad = np.nanmedian(np.abs(stimulus - median_val), axis=axis, keepdims=True)
            
            modified_z_scores = 0.6745 * (stimulus - median_val) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        cleaned_stimulus = stimulus.copy()
        cleaned_stimulus[outlier_mask] = np.nan
        
        return cleaned_stimulus, outlier_mask


class DataSynchronizer:
    """
    Utilities for synchronizing stimulus and response data.
    """
    
    @staticmethod
    def align_data(stimulus_times: np.ndarray,
                  spike_times: np.ndarray,
                  tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align stimulus and spike timing.
        
        Parameters
        ----------
        stimulus_times : numpy.ndarray
            Stimulus timestamps
        spike_times : numpy.ndarray
            Spike timestamps
        tolerance : float
            Time tolerance for alignment
            
        Returns
        -------
        aligned_stimulus_times : numpy.ndarray
            Aligned stimulus times
        aligned_spike_times : numpy.ndarray
            Aligned spike times
        """
        # Find common time range
        min_time = max(np.min(stimulus_times), np.min(spike_times))
        max_time = min(np.max(stimulus_times), np.max(spike_times))
        
        # Filter data to common range
        stim_mask = (stimulus_times >= min_time - tolerance) & \
                   (stimulus_times <= max_time + tolerance)
        spike_mask = (spike_times >= min_time - tolerance) & \
                    (spike_times <= max_time + tolerance)
        
        aligned_stimulus_times = stimulus_times[stim_mask] - min_time
        aligned_spike_times = spike_times[spike_mask] - min_time
        
        logger.info(f"Aligned data: {len(aligned_stimulus_times)} stimulus points, "
                   f"{len(aligned_spike_times)} spikes")
        
        return aligned_stimulus_times, aligned_spike_times
    
    @staticmethod
    def interpolate_stimulus(stimulus: np.ndarray,
                           old_times: np.ndarray,
                           new_times: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
        """
        Interpolate stimulus to new time points.
        
        Parameters
        ----------
        stimulus : numpy.ndarray
            Original stimulus data
        old_times : numpy.ndarray
            Original time points
        new_times : numpy.ndarray
            New time points for interpolation
        method : str
            Interpolation method: 'linear', 'cubic', 'nearest'
            
        Returns
        -------
        numpy.ndarray
            Interpolated stimulus
        """
        from scipy.interpolate import interp1d
        
        if stimulus.ndim == 1:
            interpolator = interp1d(old_times, stimulus, kind=method, 
                                  bounds_error=False, fill_value=0)
            return interpolator(new_times)
        else:
            # Handle multi-dimensional stimulus
            interpolated = np.zeros((len(new_times), stimulus.shape[1]))
            
            for i in range(stimulus.shape[1]):
                interpolator = interp1d(old_times, stimulus[:, i], kind=method,
                                      bounds_error=False, fill_value=0)
                interpolated[:, i] = interpolator(new_times)
            
            return interpolated


def validate_data_consistency(stimulus: np.ndarray,
                            spike_times: np.ndarray,
                            stimulus_times: Optional[np.ndarray] = None,
                            sampling_rate: Optional[float] = None) -> Dict[str, Any]:
    """
    Validate consistency between stimulus and spike data.
    
    Parameters
    ----------
    stimulus : numpy.ndarray
        Stimulus data
    spike_times : numpy.ndarray
        Spike times
    stimulus_times : numpy.ndarray, optional
        Stimulus timestamps
    sampling_rate : float, optional
        Stimulus sampling rate
        
    Returns
    -------
    dict
        Validation results and statistics
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    # Basic data checks
    if len(spike_times) == 0:
        results['warnings'].append("No spikes found")
    
    if stimulus.size == 0:
        results['errors'].append("Empty stimulus")
        results['valid'] = False
    
    # Check for NaN/inf values
    if np.any(np.isnan(stimulus)) or np.any(np.isinf(stimulus)):
        nan_count = np.sum(np.isnan(stimulus))
        inf_count = np.sum(np.isinf(stimulus))
        results['warnings'].append(f"Stimulus contains {nan_count} NaN and {inf_count} inf values")
    
    if np.any(np.isnan(spike_times)) or np.any(np.isinf(spike_times)):
        results['errors'].append("Spike times contain NaN or inf values")
        results['valid'] = False
    
    # Timing consistency
    if stimulus_times is not None:
        if len(stimulus_times) != len(stimulus):
            results['errors'].append("Stimulus and time arrays have different lengths")
            results['valid'] = False
        
        stim_duration = stimulus_times[-1] - stimulus_times[0]
        max_spike_time = np.max(spike_times) if len(spike_times) > 0 else 0
        
        if max_spike_time > stim_duration:
            results['warnings'].append("Spikes extend beyond stimulus duration")
    
    # Sampling rate consistency
    if sampling_rate is not None and stimulus_times is not None:
        actual_rate = 1 / np.mean(np.diff(stimulus_times))
        rate_diff = abs(actual_rate - sampling_rate) / sampling_rate
        
        if rate_diff > 0.01:  # 1% tolerance
            results['warnings'].append(f"Sampling rate mismatch: expected {sampling_rate}, "
                                     f"actual {actual_rate:.2f}")
    
    # Statistics
    results['stats'] = {
        'stimulus_shape': stimulus.shape,
        'spike_count': len(spike_times),
        'stimulus_range': [float(np.nanmin(stimulus)), float(np.nanmax(stimulus))],
        'stimulus_mean': float(np.nanmean(stimulus)),
        'stimulus_std': float(np.nanstd(stimulus))
    }
    
    if len(spike_times) > 1:
        results['stats'].update({
            'spike_rate': len(spike_times) / (np.max(spike_times) - np.min(spike_times)),
            'mean_isi': float(np.mean(np.diff(spike_times))),
            'isi_cv': float(np.std(np.diff(spike_times)) / np.mean(np.diff(spike_times)))
        })
    
    return results

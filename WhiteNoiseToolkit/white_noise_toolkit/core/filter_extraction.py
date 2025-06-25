"""
Filter extraction algorithms for the White Noise Analysis Toolkit.

This module implements streaming algorithms for extracting linear filters
from stimulus-response data using both spike-triggered average (STA) and
whitened STA (maximum likelihood) methods.
"""

import warnings
from typing import Optional, Dict, Any
import numpy as np
import scipy.linalg
from ..core.exceptions import (
    FilterExtractionError, InsufficientDataError,
    NumericalInstabilityError, DataValidationError
)


class StreamingFilterExtractor:
    """
    Streaming filter extraction with both STA and whitened STA methods.

    This class accumulates statistics across data chunks to compute linear
    filters without loading the entire dataset into memory.
    """

    def __init__(self):
        """Initialize streaming filter extractor."""
        # STA accumulators
        self.sta_accumulator: Optional[np.ndarray] = None
        self.spike_count: int = 0
        self.n_samples: int = 0

        # Whitened STA accumulators
        self.xtx_accumulator: Optional[np.ndarray] = None  # X^T * X
        self.xty_accumulator: Optional[np.ndarray] = None  # X^T * y

        # State tracking
        self.n_chunks_processed: int = 0
        self.n_features: Optional[int] = None

        # Diagnostics
        self.chunk_spike_counts: list = []
        self.chunk_sample_counts: list = []

    def compute_sta_streaming(self, design_chunk: np.ndarray,
                            spike_chunk: np.ndarray) -> None:
        """
        Accumulate STA computation for a chunk.

        The STA is computed as: STA = (X^T * y) / n_spikes

        Parameters
        ----------
        design_chunk : np.ndarray
            Design matrix chunk with shape (n_time_bins, n_features)
        spike_chunk : np.ndarray
            Spike count chunk with shape (n_time_bins,) or (n_time_bins, n_electrodes)

        Raises
        ------
        DataValidationError
            If chunk shapes are incompatible
        """
        # Validate inputs
        self._validate_chunk_shapes(design_chunk, spike_chunk)

        # Handle multi-electrode case
        if spike_chunk.ndim == 2:
            # For multi-electrode data, we'll process each electrode separately
            # For now, sum across electrodes (this can be extended later)
            spike_chunk = np.sum(spike_chunk, axis=1)

        # Ensure spike_chunk is 1D
        if spike_chunk.ndim != 1:
            raise DataValidationError(
                f"spike_chunk must be 1D or 2D, got shape {spike_chunk.shape}"
            )

        # Compute cross-correlation: X^T @ y
        cross_corr = design_chunk.T @ spike_chunk

        # Initialize or accumulate
        if self.sta_accumulator is None:
            self.sta_accumulator = cross_corr.copy()
            self.n_features = design_chunk.shape[1]
        else:
            if cross_corr.shape != self.sta_accumulator.shape:
                raise DataValidationError(
                    f"Feature dimension mismatch: expected {self.sta_accumulator.shape}, "
                    f"got {cross_corr.shape}"
                )
            self.sta_accumulator += cross_corr

        # Update counters
        chunk_spikes = int(np.sum(spike_chunk))
        chunk_samples = len(spike_chunk)

        self.spike_count += chunk_spikes
        self.n_samples += chunk_samples
        self.n_chunks_processed += 1

        # Store diagnostics
        self.chunk_spike_counts.append(chunk_spikes)
        self.chunk_sample_counts.append(chunk_samples)

    def compute_whitened_sta_streaming(self, design_chunk: np.ndarray,
                                     spike_chunk: np.ndarray) -> None:
        """
        Accumulate whitened STA computation for a chunk.

        The whitened STA is computed as: w = (X^T * X)^-1 * (X^T * y)

        Parameters
        ----------
        design_chunk : np.ndarray
            Design matrix chunk with shape (n_time_bins, n_features)
        spike_chunk : np.ndarray
            Spike count chunk with shape (n_time_bins,) or (n_time_bins, n_electrodes)

        Raises
        ------
        DataValidationError
            If chunk shapes are incompatible
        """
        # Validate inputs
        self._validate_chunk_shapes(design_chunk, spike_chunk)

        # Handle multi-electrode case
        if spike_chunk.ndim == 2:
            spike_chunk = np.sum(spike_chunk, axis=1)

        if spike_chunk.ndim != 1:
            raise DataValidationError(
                f"spike_chunk must be 1D or 2D, got shape {spike_chunk.shape}"
            )

        # Compute X^T @ X and X^T @ y
        xtx = design_chunk.T @ design_chunk
        xty = design_chunk.T @ spike_chunk

        # Initialize or accumulate
        if self.xtx_accumulator is None:
            self.xtx_accumulator = xtx.copy()
            self.xty_accumulator = xty.copy()
            self.n_features = design_chunk.shape[1]
        else:
            if xtx.shape != self.xtx_accumulator.shape:
                raise DataValidationError(
                    f"Feature dimension mismatch in X^T*X: expected {self.xtx_accumulator.shape}, "
                    f"got {xtx.shape}"
                )
            if self.xty_accumulator is not None and xty.shape != self.xty_accumulator.shape:
                raise DataValidationError(
                    f"Feature dimension mismatch in X^T*y: expected {self.xty_accumulator.shape}, "
                    f"got {xty.shape}"
                )

            self.xtx_accumulator += xtx
            if self.xty_accumulator is not None:
                self.xty_accumulator += xty

        # Update counters (same as STA)
        chunk_spikes = int(np.sum(spike_chunk))
        chunk_samples = len(spike_chunk)

        if not hasattr(self, 'spike_count'):  # In case only whitened STA is used
            self.spike_count = 0
            self.n_samples = 0
            self.n_chunks_processed = 0
            self.chunk_spike_counts = []
            self.chunk_sample_counts = []

        self.spike_count += chunk_spikes
        self.n_samples += chunk_samples
        self.n_chunks_processed += 1
        self.chunk_spike_counts.append(chunk_spikes)
        self.chunk_sample_counts.append(chunk_samples)

    def finalize_sta(self) -> np.ndarray:
        """
        Finalize STA computation.

        Returns
        -------
        np.ndarray
            Spike-triggered average filter

        Raises
        ------
        InsufficientDataError
            If no spikes were found
        FilterExtractionError
            If STA computation failed
        """
        if self.sta_accumulator is None:
            raise FilterExtractionError("No data accumulated for STA computation")

        if self.spike_count == 0:
            raise InsufficientDataError(
                "No spikes found in data - cannot compute STA",
                spike_count=0,
                min_required=1
            )

        # Compute STA
        sta = self.sta_accumulator / self.spike_count

        # Validate result
        if not np.isfinite(sta).all():
            n_nan = np.isnan(sta).sum()
            n_inf = np.isinf(sta).sum()
            raise FilterExtractionError(
                f"STA contains non-finite values: {n_nan} NaN, {n_inf} infinite"
            )

        return sta

    def finalize_whitened_sta(self, regularization: float = 1e-6) -> np.ndarray:
        """
        Finalize whitened STA computation with regularization.

        Parameters
        ----------
        regularization : float, default=1e-6
            Regularization parameter for numerical stability

        Returns
        -------
        np.ndarray
            Whitened STA filter (maximum likelihood estimate)

        Raises
        ------
        FilterExtractionError
            If whitened STA computation failed
        NumericalInstabilityError
            If matrix is ill-conditioned
        """
        if self.xtx_accumulator is None or self.xty_accumulator is None:
            raise FilterExtractionError("No data accumulated for whitened STA computation")

        # Add regularization to diagonal
        xtx_reg = self.xtx_accumulator + regularization * np.eye(self.xtx_accumulator.shape[0])

        # Check condition number
        try:
            condition_number = np.linalg.cond(xtx_reg)

            if condition_number > 1e12:
                warnings.warn(
                    f"Design matrix is ill-conditioned (condition number: {condition_number:.2e}). "
                    f"Consider increasing regularization parameter (current: {regularization:.2e}).",
                    UserWarning
                )

            if condition_number > 1e15:
                raise NumericalInstabilityError(
                    "Design matrix is severely ill-conditioned",
                    computation="whitened_sta",
                    suggestion=f"Increase regularization above {regularization:.2e}"
                )

        except np.linalg.LinAlgError:
            raise NumericalInstabilityError(
                "Cannot compute condition number - matrix may be singular",
                computation="condition_number",
                suggestion="Increase regularization or check data quality"
            )

        # Solve linear system: (X^T*X + reg*I) * w = X^T*y
        try:
            whitened_sta = scipy.linalg.solve(
                xtx_reg,
                self.xty_accumulator,
                assume_a='pos'  # Assume positive definite (faster)
            )
        except scipy.linalg.LinAlgError:
            # Fallback to general solver
            try:
                whitened_sta = scipy.linalg.solve(xtx_reg, self.xty_accumulator)
            except scipy.linalg.LinAlgError as e:
                raise FilterExtractionError(
                    f"Failed to solve linear system for whitened STA: {e}",
                    condition_number=condition_number
                )

        # Validate result
        if not np.isfinite(whitened_sta).all():
            n_nan = np.isnan(whitened_sta).sum()
            n_inf = np.isinf(whitened_sta).sum()
            raise FilterExtractionError(
                f"Whitened STA contains non-finite values: {n_nan} NaN, {n_inf} infinite"
            )

        return whitened_sta

    def _validate_chunk_shapes(self, design_chunk: np.ndarray,
                              spike_chunk: np.ndarray) -> None:
        """
        Validate that chunk shapes are compatible.

        Parameters
        ----------
        design_chunk : np.ndarray
            Design matrix chunk
        spike_chunk : np.ndarray
            Spike chunk

        Raises
        ------
        DataValidationError
            If shapes are incompatible
        """
        if design_chunk.ndim != 2:
            raise DataValidationError(
                f"design_chunk must be 2D, got shape {design_chunk.shape}"
            )

        if spike_chunk.ndim not in [1, 2]:
            raise DataValidationError(
                f"spike_chunk must be 1D or 2D, got shape {spike_chunk.shape}"
            )

        # Check temporal alignment
        if design_chunk.shape[0] != spike_chunk.shape[0]:
            raise DataValidationError(
                f"Temporal dimension mismatch: design_chunk has {design_chunk.shape[0]} "
                f"time bins, spike_chunk has {spike_chunk.shape[0]}"
            )

        # Check for empty chunks
        if design_chunk.size == 0:
            raise DataValidationError("design_chunk is empty")

        if spike_chunk.size == 0:
            raise DataValidationError("spike_chunk is empty")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics and diagnostics.

        Returns
        -------
        dict
            Dictionary with extraction statistics
        """
        stats = {
            'n_chunks_processed': self.n_chunks_processed,
            'total_spike_count': self.spike_count,
            'total_samples': self.n_samples,
            'n_features': self.n_features,
            'mean_spikes_per_chunk': np.mean(self.chunk_spike_counts) if self.chunk_spike_counts else 0,
            'std_spikes_per_chunk': np.std(self.chunk_spike_counts) if self.chunk_spike_counts else 0,
            'spike_rate_hz': None,
            'has_sta_data': self.sta_accumulator is not None,
            'has_whitened_sta_data': self.xtx_accumulator is not None,
        }

        # Compute spike rate if we have the information
        # Note: bin_size would need to be passed in or stored separately
        if self.n_samples > 0:
            # Placeholder for spike rate - would need bin_size parameter
            stats['spike_rate_hz'] = None  # self.spike_count / (self.n_samples * bin_size)
        else:
            stats['spike_rate_hz'] = None

        # Add condition number if available
        if self.xtx_accumulator is not None:
            try:
                stats['condition_number'] = np.linalg.cond(self.xtx_accumulator)
            except Exception:
                stats['condition_number'] = None

        return stats

    def reset(self) -> None:
        """Reset all accumulators for a new extraction."""
        self.sta_accumulator = None
        self.xtx_accumulator = None
        self.xty_accumulator = None
        self.spike_count = 0
        self.n_samples = 0
        self.n_chunks_processed = 0
        self.n_features = None
        self.chunk_spike_counts = []
        self.chunk_sample_counts = []

    def partial_finalize_sta(self) -> Optional[np.ndarray]:
        """
        Get current STA estimate without finalizing (for monitoring progress).

        Returns
        -------
        np.ndarray or None
            Current STA estimate, or None if no data accumulated
        """
        if self.sta_accumulator is None or self.spike_count == 0:
            return None

        return self.sta_accumulator / self.spike_count


def compare_filter_methods(sta: np.ndarray, whitened_sta: np.ndarray,
                          correlation_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Compare STA and whitened STA filters.

    Parameters
    ----------
    sta : np.ndarray
        Spike-triggered average filter
    whitened_sta : np.ndarray
        Whitened STA filter
    correlation_threshold : float, default=0.8
        Threshold for considering filters similar

    Returns
    -------
    dict
        Comparison results
    """
    if sta.shape != whitened_sta.shape:
        return {
            'correlation': None,
            'filters_similar': False,
            'error': f"Shape mismatch: STA {sta.shape}, whitened STA {whitened_sta.shape}"
        }

    # Compute correlation
    correlation = np.corrcoef(sta.flatten(), whitened_sta.flatten())[0, 1]

    # Compute relative magnitudes
    sta_norm = np.linalg.norm(sta)
    whitened_norm = np.linalg.norm(whitened_sta)
    magnitude_ratio = whitened_norm / sta_norm if sta_norm > 0 else np.inf

    # Compute SNR estimates (rough)
    sta_snr = np.max(np.abs(sta)) / np.std(sta) if np.std(sta) > 0 else 0
    whitened_snr = np.max(np.abs(whitened_sta)) / np.std(whitened_sta) if np.std(whitened_sta) > 0 else 0

    return {
        'correlation': correlation,
        'filters_similar': correlation >= correlation_threshold,
        'magnitude_ratio': magnitude_ratio,
        'sta_norm': sta_norm,
        'whitened_norm': whitened_norm,
        'sta_snr_estimate': sta_snr,
        'whitened_snr_estimate': whitened_snr,
        'snr_improvement': whitened_snr / sta_snr if sta_snr > 0 else np.inf
    }


def validate_filter_quality(filter_weights: np.ndarray,
                           filter_type: str = "unknown") -> Dict[str, Any]:
    """
    Validate extracted filter quality.

    Parameters
    ----------
    filter_weights : np.ndarray
        Extracted filter
    filter_type : str, default="unknown"
        Type of filter ("STA" or "whitened_STA")

    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'filter_type': filter_type,
        'shape': filter_weights.shape,
        'is_finite': np.isfinite(filter_weights).all(),
        'is_zero': np.allclose(filter_weights, 0),
        'norm': np.linalg.norm(filter_weights),
        'max_abs_value': np.max(np.abs(filter_weights)),
        'warnings': []
    }

    # Check for problems
    if not validation['is_finite']:
        n_nan = np.isnan(filter_weights).sum()
        n_inf = np.isinf(filter_weights).sum()
        validation['warnings'].append(f"Non-finite values: {n_nan} NaN, {n_inf} infinite")

    if validation['is_zero']:
        validation['warnings'].append("Filter is effectively zero - may indicate insufficient data")

    if validation['norm'] < 1e-10:
        validation['warnings'].append("Filter has very small norm - may be unreliable")

    if validation['max_abs_value'] > 1e6:
        validation['warnings'].append("Filter has very large values - may indicate numerical issues")

    # Filter-specific checks
    if len(filter_weights) > 1:
        # Check for reasonable temporal structure
        autocorr = np.correlate(filter_weights, filter_weights, mode='full')
        central_idx = len(autocorr) // 2
        normalized_autocorr = autocorr / autocorr[central_idx]

        # Check if filter is too smooth or too noisy
        if len(filter_weights) > 5:
            differences = np.diff(filter_weights)
            smoothness = np.std(differences) / np.std(filter_weights) if np.std(filter_weights) > 0 else 0

            if smoothness < 0.01:
                validation['warnings'].append("Filter may be too smooth")
            elif smoothness > 2.0:
                validation['warnings'].append("Filter may be too noisy")

            validation['smoothness'] = smoothness

    return validation

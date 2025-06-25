"""
Single cell analyzer for the White Noise Analysis Toolkit.

This module provides the main SingleCellAnalyzer class that integrates all
components for complete white noise analysis of individual neurons.
"""

import warnings
from typing import Optional, Dict, Any, Generator, Tuple, Union, List
import numpy as np
from tqdm import tqdm

from .design_matrix import StreamingDesignMatrix, create_design_matrix_batch
from .filter_extraction import StreamingFilterExtractor, validate_filter_quality, compare_filter_methods
from .nonlinearity_estimation import NonparametricNonlinearity, ParametricNonlinearity
from .streaming_analyzer import validate_generators, estimate_total_chunks
from ..utils.memory_manager import MemoryManager
from ..utils.logging_config import get_logger, TimingLogger, MemoryLogger, ProgressLogger
from ..core.exceptions import (
    FilterExtractionError, DataValidationError, InsufficientDataError,
    MemoryLimitError, NonlinearityFittingError
)


class SingleCellAnalyzer:
    """
    Complete white noise analysis for a single neuron.

    This class integrates stimulus design matrix construction, filter extraction,
    and nonlinearity estimation in a streaming framework for memory efficiency.
    """

    def __init__(self, bin_size: float = 0.008, filter_length: int = 25,
                 spatial_dims: Optional[Tuple[int, ...]] = None, n_colors: int = 1,
                 memory_limit_gb: float = 8.0):
        """
        Initialize single-cell white noise analyzer.

        Parameters
        ----------
        bin_size : float, default=0.008
            Temporal bin size in seconds
        filter_length : int, default=25
            Number of time bins for filter
        spatial_dims : tuple, optional
            (height, width) for spatial stimuli
        n_colors : int, default=1
            Number of color channels
        memory_limit_gb : float, default=8.0
            Memory limit for analysis

        Raises
        ------
        DataValidationError
            If parameters are invalid
        """
        # Validate parameters
        if bin_size <= 0:
            raise DataValidationError(f"bin_size must be positive, got {bin_size}")

        if filter_length <= 0:
            raise DataValidationError(f"filter_length must be positive, got {filter_length}")

        if n_colors <= 0:
            raise DataValidationError(f"n_colors must be positive, got {n_colors}")

        if memory_limit_gb <= 0:
            raise DataValidationError(f"memory_limit_gb must be positive, got {memory_limit_gb}")

        # Store parameters
        self.bin_size = bin_size
        self.filter_length = filter_length
        self.spatial_dims = spatial_dims
        self.n_colors = n_colors
        self.memory_limit_gb = memory_limit_gb

        # Initialize components
        self.design_matrix_builder = StreamingDesignMatrix(
            filter_length, spatial_dims, n_colors)
        self.filter_extractor = StreamingFilterExtractor()
        self.memory_manager = MemoryManager(max_memory_gb=memory_limit_gb)

        # Get logger
        self.logger = get_logger(__name__)
        self.memory_logger = MemoryLogger(self.logger, self.memory_manager)

        # Results storage
        self.filter_w: Optional[np.ndarray] = None
        self.sta: Optional[np.ndarray] = None
        self.nonlinearity_N: Optional[Union[NonparametricNonlinearity, ParametricNonlinearity]] = None
        self.performance_metrics: Dict[str, Any] = {}
        self.analysis_metadata: Dict[str, Any] = {}
        self.fitted = False

        # Analysis state
        self._total_spikes = 0
        self._total_samples = 0
        self._n_chunks_processed = 0

    def fit_streaming(self, stimulus_generator: Generator[np.ndarray, None, None],
                     spike_generator: Generator[np.ndarray, None, None],
                     chunk_size: int = 1000,
                     nonlinearity_method: str = 'nonparametric',
                     extract_both_filters: bool = True,
                     progress_bar: bool = True,
                     **kwargs) -> None:
        """
        Main analysis pipeline using streaming computation.

        This method performs the complete analysis in two passes:
        Pass 1: Extract linear filter(s)
        Pass 2: Estimate nonlinearity

        Parameters
        ----------
        stimulus_generator : Generator
            Yields stimulus chunks
        spike_generator : Generator
            Yields spike chunks
        chunk_size : int, default=1000
            Chunk size for streaming
        nonlinearity_method : str, default='nonparametric'
            'nonparametric' or 'parametric'
        extract_both_filters : bool, default=True
            Whether to extract both STA and whitened STA
        progress_bar : bool, default=True
            Whether to show progress bars
        **kwargs
            Additional parameters for nonlinearity estimation

        Raises
        ------
        FilterExtractionError
            If filter extraction fails
        NonlinearityFittingError
            If nonlinearity estimation fails
        """
        self.logger.info("Starting streaming white noise analysis")
        self.memory_logger.log_memory_usage("analysis start")

        # Reset state
        self.reset()

        # Store analysis parameters
        self.analysis_metadata.update({
            'chunk_size': chunk_size,
            'nonlinearity_method': nonlinearity_method,
            'extract_both_filters': extract_both_filters,
            'bin_size': self.bin_size,
            'filter_length': self.filter_length,
            'spatial_dims': self.spatial_dims,
            'n_colors': self.n_colors
        })

        try:
            # Collect data from generators for multiple processing passes
            stimulus_chunks = []
            spike_chunks = []

            self.logger.info("Collecting data from generators...")
            for stim_chunk, spike_chunk in zip(stimulus_generator, spike_generator):
                if stim_chunk.size > 0 and spike_chunk.size > 0:
                    stimulus_chunks.append(stim_chunk)
                    spike_chunks.append(spike_chunk)

            if not stimulus_chunks:
                raise InsufficientDataError("No data received from generators")

            self.logger.info(f"Collected {len(stimulus_chunks)} chunks")

            # Pass 1: Extract filter(s)
            with TimingLogger(self.logger, "filter extraction"):
                self._extract_filter_from_chunks(
                    stimulus_chunks, spike_chunks, extract_both_filters, progress_bar
                )

            self.memory_logger.log_memory_usage("after filter extraction")

            # Pass 2: Estimate nonlinearity (reuse the same data)
            with TimingLogger(self.logger, "nonlinearity estimation"):
                self._estimate_nonlinearity_from_chunks(
                    stimulus_chunks, spike_chunks, nonlinearity_method, progress_bar, **kwargs
                )

            self.memory_logger.log_memory_usage("after nonlinearity estimation")

            # Compute performance metrics
            with TimingLogger(self.logger, "performance metrics"):
                self._compute_performance_metrics()

            self.fitted = True
            self.logger.info("Analysis completed successfully")

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def _extract_filter_streaming(self, stimulus_generator: Generator,
                                 spike_generator: Generator,
                                 chunk_size: int,
                                 extract_both: bool,
                                 progress_bar: bool) -> None:
        """Extract filter(s) using streaming computation."""
        self.logger.info("Extracting linear filter(s)")

        # Estimate total chunks for progress tracking
        try:
            # This is an approximation since we can't know exact count without consuming generators
            estimated_chunks = 100  # Default estimate
        except Exception:
            estimated_chunks = 100

        # Set up progress tracking
        pbar = None
        if progress_bar:
            pbar = tqdm(desc="Filter extraction", unit="chunks", leave=False)

        progress_logger = ProgressLogger(
            self.logger, estimated_chunks, "Filter extraction",
            memory_manager=self.memory_manager
        )

        chunk_count = 0
        is_first_chunk = True

        try:
            # Process chunks
            for stim_chunk, spike_chunk in zip(stimulus_generator, spike_generator):
                # Validate chunk
                if stim_chunk.size == 0 or spike_chunk.size == 0:
                    continue

                # Build design matrix for this chunk
                design_chunk = self.design_matrix_builder.build_hankel_chunk(
                    stim_chunk, is_first_chunk=is_first_chunk
                )

                # Extract filters
                self.filter_extractor.compute_sta_streaming(design_chunk, spike_chunk)

                if extract_both:
                    self.filter_extractor.compute_whitened_sta_streaming(design_chunk, spike_chunk)

                # Update counters
                chunk_count += 1
                self._n_chunks_processed += 1
                self._total_spikes += int(np.sum(spike_chunk))
                self._total_samples += len(spike_chunk)

                # Update progress
                if progress_bar and pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        'spikes': self._total_spikes,
                        'memory': f"{self.memory_manager.get_current_usage():.1f}GB"
                    })

                progress_logger.update(1)

                # Check memory usage
                self.memory_manager.check_memory_warning()

                is_first_chunk = False

        finally:
            if progress_bar and pbar is not None:
                pbar.close()

        # Finalize filters
        self.logger.info(f"Processed {chunk_count} chunks, {self._total_spikes} total spikes")

        if self._total_spikes == 0:
            raise InsufficientDataError(
                "No spikes found in data",
                spike_count=0,
                min_required=1
            )

        # Extract STA
        self.sta = self.filter_extractor.finalize_sta()
        self.logger.info(f"Extracted STA with norm: {np.linalg.norm(self.sta):.4f}")

        # Extract whitened STA if requested
        if extract_both:
            try:
                self.filter_w = self.filter_extractor.finalize_whitened_sta()
                self.logger.info(f"Extracted whitened STA with norm: {np.linalg.norm(self.filter_w):.4f}")

                # Compare filters
                comparison = compare_filter_methods(self.sta, self.filter_w)
                self.logger.info(f"Filter correlation: {comparison['correlation']:.3f}")
                self.analysis_metadata['filter_comparison'] = comparison

            except Exception as e:
                self.logger.warning(f"Whitened STA extraction failed: {e}")
                self.filter_w = self.sta.copy()  # Fallback to STA
        else:
            self.filter_w = self.sta.copy()

        # Validate filter quality
        filter_validation = validate_filter_quality(self.filter_w, "whitened_STA")
        if filter_validation['warnings']:
            for warning in filter_validation['warnings']:
                self.logger.warning(f"Filter quality: {warning}")

        self.analysis_metadata['filter_validation'] = filter_validation

    def _estimate_nonlinearity_streaming(self, stimulus_generator: Generator,
                                       spike_generator: Generator,
                                       chunk_size: int,
                                       method: str,
                                       progress_bar: bool,
                                       **kwargs) -> None:
        """Estimate nonlinearity using streaming computation."""
        self.logger.info(f"Estimating nonlinearity using {method} method")

        if self.filter_w is None:
            raise FilterExtractionError("Filter not extracted - cannot estimate nonlinearity")

        # Collect generator signals and spike counts
        generator_signals = []
        spike_counts = []

        pbar = None
        if progress_bar:
            pbar = tqdm(desc="Nonlinearity estimation", unit="chunks", leave=False)

        is_first_chunk = True
        chunk_count = 0

        try:
            for stim_chunk, spike_chunk in zip(stimulus_generator, spike_generator):
                if stim_chunk.size == 0 or spike_chunk.size == 0:
                    continue

                # Build design matrix
                design_chunk = self.design_matrix_builder.build_hankel_chunk(
                    stim_chunk, is_first_chunk=is_first_chunk
                )

                # Compute generator signal
                generator_signal = design_chunk @ self.filter_w

                # Handle multi-electrode spike data
                if spike_chunk.ndim > 1:
                    spike_chunk = np.sum(spike_chunk, axis=1)

                # Store for nonlinearity fitting
                generator_signals.append(generator_signal)
                spike_counts.append(spike_chunk)

                chunk_count += 1
                if progress_bar and pbar is not None:
                    pbar.update(1)

                is_first_chunk = False

        finally:
            if progress_bar and pbar is not None:
                pbar.close()

        if not generator_signals:
            raise InsufficientDataError("No data collected for nonlinearity estimation")

        # Concatenate all data
        full_generator_signal = np.concatenate(generator_signals)
        full_spike_counts = np.concatenate(spike_counts)

        self.logger.info(f"Collected {len(full_generator_signal)} samples for nonlinearity fitting")

        # Fit nonlinearity
        if method == 'nonparametric':
            n_bins = kwargs.get('n_bins', 25)
            self.nonlinearity_N = NonparametricNonlinearity(n_bins=n_bins)
        elif method == 'parametric':
            model_type = kwargs.get('model_type', 'exponential')
            self.nonlinearity_N = ParametricNonlinearity(model_type=model_type)
        else:
            raise DataValidationError(f"Unknown nonlinearity method: {method}")

        try:
            self.nonlinearity_N.fit(full_generator_signal, full_spike_counts)
            self.logger.info(f"Successfully fitted {method} nonlinearity")

            # Compute goodness of fit
            if isinstance(self.nonlinearity_N, NonparametricNonlinearity):
                gof = self.nonlinearity_N.get_goodness_of_fit(full_generator_signal, full_spike_counts)
                self.logger.info(f"Nonlinearity R²: {gof.get('r_squared', 'N/A'):.3f}")
                self.analysis_metadata['nonlinearity_gof'] = gof

        except Exception as e:
            raise NonlinearityFittingError(f"Nonlinearity fitting failed: {e}", fitting_method=method)

    def _compute_performance_metrics(self) -> None:
        """Compute performance metrics for the fitted model."""
        if not self.fitted or self.filter_w is None or self.nonlinearity_N is None:
            return

        # Basic statistics
        self.performance_metrics.update({
            'total_spikes': self._total_spikes,
            'total_samples': self._total_samples,
            'n_chunks_processed': self._n_chunks_processed,
            'spike_rate_hz': self._total_spikes / (self._total_samples * self.bin_size),
            'filter_norm': np.linalg.norm(self.filter_w),
            'filter_snr_estimate': np.max(np.abs(self.filter_w)) / np.std(self.filter_w)
        })

        # Extract filter statistics
        extractor_stats = self.filter_extractor.get_statistics()
        self.performance_metrics.update({
            'filter_extraction': extractor_stats
        })

        # Memory usage
        memory_info = self.memory_manager.get_memory_info()
        self.performance_metrics.update({
            'memory_usage': memory_info
        })

    def predict(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Use extracted filter and nonlinearity to predict spike rates.

        Parameters
        ----------
        stimulus : np.ndarray
            Stimulus array

        Returns
        -------
        np.ndarray
            Predicted spike rates

        Raises
        ------
        FilterExtractionError
            If model not fitted
        """
        if not self.fitted:
            raise FilterExtractionError("Model not fitted. Call fit_streaming() first.")

        if self.filter_w is None or self.nonlinearity_N is None:
            raise FilterExtractionError("Filter or nonlinearity not available")

        # Build design matrix
        design_matrix = self.design_matrix_builder.build_hankel_chunk(
            stimulus, is_first_chunk=True
        )

        # Compute generator signal
        generator_signal = design_matrix @ self.filter_w

        # Apply nonlinearity
        return self.nonlinearity_N.predict(generator_signal)

    def get_results(self) -> Dict[str, Any]:
        """
        Return structured results dictionary.

        Returns
        -------
        dict
            Complete analysis results

        Raises
        ------
        FilterExtractionError
            If model not fitted
        """
        if not self.fitted:
            raise FilterExtractionError("Model not fitted. Call fit_streaming() first.")

        results: Dict[str, Any] = {
            'filter': self.filter_w,
            'sta': self.sta
        }

        # Add nonlinearity if available
        if hasattr(self, 'nonlinearity_data') and self.nonlinearity_data is not None:
            results['nonlinearity'] = self.nonlinearity_data

        # Add performance metrics if available
        if hasattr(self, 'performance_metrics') and self.performance_metrics is not None:
            results['performance_metrics'] = self.performance_metrics

        # Add analysis metadata
        results['metadata'] = self.analysis_metadata

        return results

    def _extract_temporal_profile(self) -> Optional[np.ndarray]:
        """Extract temporal filter profile."""
        if self.filter_w is None:
            return None

        if self.spatial_dims is None:
            # Pure temporal filter
            return self.filter_w.copy()
        else:
            # Spatial-temporal: extract temporal component
            # Reshape filter: (filter_length, spatial_size)
            filter_reshaped = self.filter_w.reshape(self.filter_length, -1)
            # Take mean across spatial dimensions as temporal profile
            return np.mean(filter_reshaped, axis=1)

    def _extract_spatial_profile(self) -> Optional[np.ndarray]:
        """Extract spatial filter profile."""
        if self.filter_w is None or self.spatial_dims is None:
            return None

        # Reshape filter: (filter_length, height, width, n_colors)
        if self.n_colors == 1:
            filter_reshaped = self.filter_w.reshape(
                self.filter_length, self.spatial_dims[0], self.spatial_dims[1]
            )
        else:
            filter_reshaped = self.filter_w.reshape(
                self.filter_length, self.spatial_dims[0], self.spatial_dims[1], self.n_colors
            )

        # Take mean across time as spatial profile
        return np.mean(filter_reshaped, axis=0)

    def reset(self) -> None:
        """Reset analyzer state for new analysis."""
        self.filter_w = None
        self.sta = None
        self.nonlinearity_N = None
        self.performance_metrics = {}
        self.analysis_metadata = {}
        self.fitted = False

        self._total_spikes = 0
        self._total_samples = 0
        self._n_chunks_processed = 0

        # Reset components
        self.filter_extractor.reset()
        self.design_matrix_builder.reset_buffer()

        self.logger.debug("Analyzer state reset")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        return self.memory_manager.get_memory_info()

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration.

        Returns
        -------
        dict
            Validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }

        # Check memory requirements
        try:
            # Estimate memory for typical chunk
            typical_chunk_shape = (1000,)
            if self.spatial_dims:
                typical_chunk_shape = (1000,) + self.spatial_dims
                if self.n_colors > 1:
                    typical_chunk_shape = typical_chunk_shape + (self.n_colors,)

            self.memory_manager.validate_memory_requirements(
                typical_chunk_shape, np.dtype(np.float64), self.filter_length
            )

        except MemoryLimitError as e:
            validation['errors'].append(str(e))
            validation['valid'] = False

        # Check parameter ranges
        if self.filter_length > 100:
            validation['warnings'].append(
                f"Large filter length ({self.filter_length}) may require significant memory"
            )

        if self.spatial_dims and np.prod(self.spatial_dims) > 10000:
            validation['warnings'].append(
                f"Large spatial dimensions {self.spatial_dims} may require significant memory"
            )

        return validation

    def _extract_filter_from_chunks(self, stimulus_chunks: List[np.ndarray],
                                   spike_chunks: List[np.ndarray],
                                   extract_both: bool,
                                   progress_bar: bool) -> None:
        """Extract filter(s) from pre-collected data chunks."""
        self.logger.info("Starting filter extraction...")

        # Initialize filter extractor
        first_stim = stimulus_chunks[0]
        first_spikes = spike_chunks[0]

        # Handle multi-electrode spike data
        if first_spikes.ndim > 1:
            first_spikes = np.sum(first_spikes, axis=1)

        # Initialize based on first chunk
        try:
            self.filter_extractor = StreamingFilterExtractor()
        except Exception as e:
            raise FilterExtractionError(f"Failed to initialize filter extractor: {e}")

        # Process all chunks
        pbar = None
        if progress_bar:
            pbar = tqdm(total=len(stimulus_chunks), desc="Filter extraction", unit="chunks")

        try:
            total_spikes = 0

            for i, (stim_chunk, spike_chunk) in enumerate(zip(stimulus_chunks, spike_chunks)):
                if stim_chunk.size == 0 or spike_chunk.size == 0:
                    continue

                # Handle multi-electrode spike data
                if spike_chunk.ndim > 1:
                    spike_chunk = np.sum(spike_chunk, axis=1)

                # Create design matrix for this chunk
                design_matrix = create_design_matrix_batch(
                    stim_chunk,
                    filter_length=self.filter_length,
                    spatial_dims=self.spatial_dims,
                    n_colors=self.n_colors
                )

                # Update filter extractor with design matrix and spikes
                self.filter_extractor.compute_sta_streaming(design_matrix, spike_chunk)
                if extract_both:
                    self.filter_extractor.compute_whitened_sta_streaming(design_matrix, spike_chunk)

                total_spikes += np.sum(spike_chunk)

                if progress_bar and pbar is not None:
                    pbar.update(1)

            if progress_bar and pbar is not None:
                pbar.close()

            self.logger.info(f"Processed {len(stimulus_chunks)} chunks, {total_spikes} total spikes")

            if total_spikes == 0:
                raise InsufficientDataError(f"No spikes found in data\nDetails: Found {total_spikes} spikes, need at least 1")

            # Extract STA
            self.sta = self.filter_extractor.finalize_sta()
            self.logger.info(f"Extracted STA with norm: {np.linalg.norm(self.sta):.4f}")

            # Extract whitened STA if requested
            if extract_both:
                try:
                    self.filter_w = self.filter_extractor.finalize_whitened_sta()
                    self.logger.info(f"Extracted whitened STA with norm: {np.linalg.norm(self.filter_w):.4f}")

                    # Compare filters
                    comparison = compare_filter_methods(self.sta, self.filter_w)
                    self.logger.info(f"Filter correlation: {comparison['correlation']:.3f}")
                    self.analysis_metadata['filter_comparison'] = comparison

                except Exception as e:
                    self.logger.warning(f"Whitened STA extraction failed: {e}")
                    self.filter_w = self.sta.copy()  # Fallback to STA
            else:
                self.filter_w = self.sta.copy()

            # Validate filter quality
            filter_validation = validate_filter_quality(self.filter_w, "whitened_STA")
            if filter_validation['warnings']:
                for warning in filter_validation['warnings']:
                    self.logger.warning(f"Filter quality: {warning}")

            self.analysis_metadata['filter_validation'] = filter_validation

        except Exception as e:
            if progress_bar and pbar is not None:
                pbar.close()
            raise FilterExtractionError(f"Filter extraction failed: {e}")

    def _estimate_nonlinearity_from_chunks(self, stimulus_chunks: List[np.ndarray],
                                          spike_chunks: List[np.ndarray],
                                          method: str,
                                          progress_bar: bool,
                                          **kwargs) -> None:
        """Estimate nonlinearity from pre-collected data chunks."""
        self.logger.info(f"Estimating nonlinearity using {method} method")

        if self.filter_w is None:
            raise FilterExtractionError("Filter not extracted - cannot estimate nonlinearity")

        # Collect generator signals and spike counts
        generator_signals = []
        spike_counts = []

        pbar = None
        if progress_bar:
            pbar = tqdm(total=len(stimulus_chunks), desc="Nonlinearity estimation", unit="chunks")

        try:
            for stim_chunk, spike_chunk in zip(stimulus_chunks, spike_chunks):
                if stim_chunk.size == 0 or spike_chunk.size == 0:
                    continue

                # Create design matrix for this chunk
                design_matrix = create_design_matrix_batch(
                    stim_chunk,
                    filter_length=self.filter_length,
                    spatial_dims=self.spatial_dims,
                    n_colors=self.n_colors
                )

                # Compute generator signal
                generator_signal = design_matrix @ self.filter_w

                # Handle multi-electrode spike data
                if spike_chunk.ndim > 1:
                    spike_chunk = np.sum(spike_chunk, axis=1)

                # Store for nonlinearity fitting
                generator_signals.append(generator_signal)
                spike_counts.append(spike_chunk)

                if progress_bar and pbar is not None:
                    pbar.update(1)

        finally:
            if progress_bar and pbar is not None:
                pbar.close()

        if not generator_signals:
            raise InsufficientDataError("No data collected for nonlinearity estimation")

        # Concatenate all data
        full_generator_signal = np.concatenate(generator_signals)
        full_spike_counts = np.concatenate(spike_counts)

        self.logger.info(f"Collected {len(full_generator_signal)} samples for nonlinearity fitting")

        # Fit nonlinearity
        if method == 'nonparametric':
            n_bins = kwargs.get('n_bins', 25)

            # Initialize nonlinearity estimator
            self.nonlinearity_N = NonparametricNonlinearity(n_bins=n_bins)

            # Fit the nonlinearity
            self.nonlinearity_N.fit(full_generator_signal, full_spike_counts)

            # Extract nonlinearity data from fitted estimator
            if self.nonlinearity_N.bin_centers is not None:
                x_values = self.nonlinearity_N.bin_centers
                y_values = self.nonlinearity_N.spike_rates

                self.nonlinearity_data = {
                    'x_values': x_values,
                    'y_values': y_values,
                    'method': 'nonparametric',
                    'n_bins': n_bins
                }

                self.logger.info(f"Fitted nonparametric nonlinearity with {len(x_values)} bins")
            else:
                raise NonlinearityFittingError("Failed to fit nonparametric nonlinearity")

        elif method == 'parametric':
            model_type = kwargs.get('model_type', 'exponential')

            # Initialize parametric nonlinearity estimator
            self.nonlinearity_N = ParametricNonlinearity(model_type=model_type)

            # Fit the nonlinearity
            self.nonlinearity_N.fit(full_generator_signal, full_spike_counts)

            # Get parameters and generate curve
            params = self.nonlinearity_N.parameters
            x_range = np.linspace(
                np.min(full_generator_signal),
                np.max(full_generator_signal),
                100
            )
            y_pred = self.nonlinearity_N.predict(x_range)

            self.nonlinearity_data = {
                'x_values': x_range,
                'y_values': y_pred,
                'parameters': params,
                'method': 'parametric',
                'model_type': model_type
            }

            self.logger.info(f"Fitted {model_type} nonlinearity: {params}")

        else:
            raise ValueError(f"Unknown nonlinearity method: {method}")

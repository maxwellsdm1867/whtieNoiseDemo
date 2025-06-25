"""
Design matrix construction for the White Noise Analysis Toolkit.

This module implements streaming Hankel matrix construction for both temporal
and spatial-temporal stimuli, with proper handling of temporal continuity
across chunks.
"""

import warnings
from typing import Tuple, Optional, Union
import numpy as np
from ..core.exceptions import DataValidationError, MemoryLimitError


class StreamingDesignMatrix:
    """
    Streaming design matrix builder with temporal continuity.
    
    This class constructs design matrices (Hankel matrices) for stimulus data
    in a streaming fashion, maintaining proper temporal continuity across chunks.
    """
    
    def __init__(self, filter_length: int, spatial_dims: Optional[Tuple[int, ...]] = None, 
                 n_colors: int = 1):
        """
        Initialize streaming design matrix builder.
        
        Parameters
        ----------
        filter_length : int
            Number of time bins for the filter (e.g., 25)
        spatial_dims : tuple, optional
            (height, width) for spatial stimuli, None for temporal-only
        n_colors : int, default=1
            Number of color channels (1 for monochrome, 3 for RGB)
            
        Raises
        ------
        DataValidationError
            If parameters are invalid
        """
        # Validate inputs
        if filter_length <= 0:
            raise DataValidationError(
                f"filter_length must be positive, got {filter_length}"
            )
        
        if spatial_dims is not None:
            if len(spatial_dims) != 2:
                raise DataValidationError(
                    f"spatial_dims must be a 2-tuple (height, width), got {spatial_dims}"
                )
            if any(d <= 0 for d in spatial_dims):
                raise DataValidationError(
                    f"spatial_dims must contain positive values, got {spatial_dims}"
                )
        
        if n_colors <= 0:
            raise DataValidationError(
                f"n_colors must be positive, got {n_colors}"
            )
        
        self.filter_length = filter_length
        self.spatial_dims = spatial_dims
        self.n_colors = n_colors
        self.buffer = None  # Store filter_length-1 samples for continuity
        
        self._initialize_dimensions()
    
    def _initialize_dimensions(self) -> None:
        """Initialize dimension calculations."""
        if self.spatial_dims is None:
            # Temporal-only stimulus
            self.spatial_size = 1
            self.n_features = self.filter_length
        else:
            # Spatial-temporal stimulus
            self.spatial_size = np.prod(self.spatial_dims) * self.n_colors
            self.n_features = self.filter_length * self.spatial_size
    
    def build_hankel_chunk(self, stimulus_chunk: np.ndarray, 
                          is_first_chunk: bool = False) -> np.ndarray:
        """
        Build design matrix chunk with proper temporal continuity.
        
        Parameters
        ----------
        stimulus_chunk : np.ndarray
            Input stimulus chunk with shape:
            - 1D temporal: (n_time_bins,)
            - 2D spatial: (n_time_bins, height, width)
            - 3D with colors: (n_time_bins, height, width, n_colors)
        is_first_chunk : bool, default=False
            Whether this is the first chunk (use zero padding)
            
        Returns
        -------
        np.ndarray
            Design matrix with shape (n_time_bins, n_features)
            where n_features = filter_length * spatial_size * n_colors
            
        Raises
        ------
        DataValidationError
            If stimulus chunk shape is invalid
        """
        # Validate input shape
        self._validate_input_shape(stimulus_chunk)
        
        # Handle temporal continuity
        padded_stimulus = self._handle_temporal_continuity(stimulus_chunk, is_first_chunk)
        
        # Flatten spatial dimensions while preserving temporal structure
        if padded_stimulus.ndim > 1:
            # Reshape to (n_time_bins, spatial_features)
            n_time_bins = padded_stimulus.shape[0]
            spatial_features = np.prod(padded_stimulus.shape[1:])
            padded_stimulus_flat = padded_stimulus.reshape(n_time_bins, spatial_features)
        else:
            # Already 1D temporal
            padded_stimulus_flat = padded_stimulus.reshape(-1, 1)
        
        # Build Hankel matrix using sliding window
        try:
            # Use numpy's sliding_window_view for efficient Hankel construction
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                # Modern numpy version
                hankel_matrix = sliding_window_view(
                    padded_stimulus_flat, 
                    window_shape=(self.filter_length,),
                    axis=0
                )
                # Shape: (n_time_bins, spatial_features, filter_length)
                # Reshape to: (n_time_bins, filter_length * spatial_features)
                design_matrix = hankel_matrix.reshape(
                    stimulus_chunk.shape[0], 
                    self.n_features
                )
            except ImportError:
                # Fallback for older numpy versions
                design_matrix = self._build_hankel_manual(
                    padded_stimulus_flat, stimulus_chunk.shape[0]
                )
        
        except Exception as e:
            raise DataValidationError(f"Failed to build design matrix: {e}")
        
        # Update buffer for next chunk
        self._update_buffer(stimulus_chunk)
        
        return design_matrix
    
    def _build_hankel_manual(self, padded_stimulus_flat: np.ndarray, 
                           n_output_bins: int) -> np.ndarray:
        """
        Manual Hankel matrix construction for older numpy versions.
        
        Parameters
        ----------
        padded_stimulus_flat : np.ndarray
            Flattened padded stimulus
        n_output_bins : int
            Number of output time bins
            
        Returns
        -------
        np.ndarray
            Design matrix
        """
        n_spatial_features = padded_stimulus_flat.shape[1]
        design_matrix = np.zeros((n_output_bins, self.n_features))
        
        for t in range(n_output_bins):
            for lag in range(self.filter_length):
                time_idx = t + lag
                start_col = lag * n_spatial_features
                end_col = (lag + 1) * n_spatial_features
                design_matrix[t, start_col:end_col] = padded_stimulus_flat[time_idx, :]
        
        return design_matrix
    
    def _handle_temporal_continuity(self, stimulus_chunk: np.ndarray, 
                                  is_first_chunk: bool) -> np.ndarray:
        """
        Handle temporal continuity between chunks using buffer.
        
        Parameters
        ----------
        stimulus_chunk : np.ndarray
            Current stimulus chunk
        is_first_chunk : bool
            Whether this is the first chunk
            
        Returns
        -------
        np.ndarray
            Padded stimulus with proper continuity
        """
        if is_first_chunk:
            # For first chunk, pad with zeros
            if stimulus_chunk.ndim == 1:
                # 1D temporal
                zeros = np.zeros(self.filter_length - 1)
                padded_stimulus = np.concatenate([zeros, stimulus_chunk], axis=0)
            else:
                # Multi-dimensional
                zero_shape = (self.filter_length - 1,) + stimulus_chunk.shape[1:]
                zeros = np.zeros(zero_shape)
                padded_stimulus = np.concatenate([zeros, stimulus_chunk], axis=0)
        else:
            # For subsequent chunks, use buffer for continuity
            if self.buffer is None:
                warnings.warn(
                    "Buffer is None for non-first chunk. Using zero padding.",
                    UserWarning
                )
                # Fallback to zero padding
                if stimulus_chunk.ndim == 1:
                    zeros = np.zeros(self.filter_length - 1)
                    padded_stimulus = np.concatenate([zeros, stimulus_chunk], axis=0)
                else:
                    zero_shape = (self.filter_length - 1,) + stimulus_chunk.shape[1:]
                    zeros = np.zeros(zero_shape)
                    padded_stimulus = np.concatenate([zeros, stimulus_chunk], axis=0)
            else:
                # Use buffer from previous chunk
                padded_stimulus = np.concatenate([self.buffer, stimulus_chunk], axis=0)
        
        return padded_stimulus
    
    def _update_buffer(self, stimulus_chunk: np.ndarray) -> None:
        """
        Update buffer with last filter_length-1 samples.
        
        Parameters
        ----------
        stimulus_chunk : np.ndarray
            Current stimulus chunk
        """
        if stimulus_chunk.shape[0] >= self.filter_length - 1:
            # Take last filter_length-1 samples
            self.buffer = stimulus_chunk[-(self.filter_length - 1):].copy()
        else:
            # Chunk is smaller than buffer size - this is unusual but handle it
            if self.buffer is None:
                self.buffer = stimulus_chunk.copy()
            else:
                # Concatenate with existing buffer and keep last filter_length-1
                combined = np.concatenate([self.buffer, stimulus_chunk], axis=0)
                if combined.shape[0] >= self.filter_length - 1:
                    self.buffer = combined[-(self.filter_length - 1):].copy()
                else:
                    self.buffer = combined.copy()
    
    def _validate_input_shape(self, stimulus_chunk: np.ndarray) -> None:
        """
        Validate input stimulus chunk shape.
        
        Parameters
        ----------
        stimulus_chunk : np.ndarray
            Stimulus chunk to validate
            
        Raises
        ------
        DataValidationError
            If shape is invalid
        """
        if stimulus_chunk.size == 0:
            raise DataValidationError("Stimulus chunk is empty")
        
        if stimulus_chunk.ndim == 0:
            raise DataValidationError("Stimulus chunk has no dimensions")
        
        # Check if shape is consistent with expected dimensions
        if self.spatial_dims is None:
            # Expecting 1D temporal stimulus
            if stimulus_chunk.ndim != 1:
                raise DataValidationError(
                    f"Expected 1D temporal stimulus, got shape {stimulus_chunk.shape}. "
                    f"Initialize with spatial_dims if using spatial stimulus."
                )
        else:
            # Expecting spatial-temporal stimulus
            expected_ndim = 3 if self.n_colors == 1 else 4
            expected_shape = (stimulus_chunk.shape[0],) + self.spatial_dims
            if self.n_colors > 1:
                expected_shape = expected_shape + (self.n_colors,)
            
            if stimulus_chunk.ndim != expected_ndim:
                raise DataValidationError(
                    f"Expected {expected_ndim}D spatial-temporal stimulus, "
                    f"got {stimulus_chunk.ndim}D with shape {stimulus_chunk.shape}"
                )
            
            # Check spatial dimensions match
            if stimulus_chunk.shape[1:] != expected_shape[1:]:
                raise DataValidationError(
                    f"Expected spatial shape {expected_shape[1:]}, "
                    f"got {stimulus_chunk.shape[1:]}"
                )
    
    def reset_buffer(self) -> None:
        """Reset the buffer for a new streaming session."""
        self.buffer = None
    
    def get_feature_count(self) -> int:
        """
        Get the number of features in the design matrix.
        
        Returns
        -------
        int
            Number of features (filter_length * spatial_size * n_colors)
        """
        return self.n_features
    
    def get_buffer_info(self) -> dict:
        """
        Get information about the current buffer state.
        
        Returns
        -------
        dict
            Buffer information
        """
        return {
            'buffer_size': self.buffer.shape if self.buffer is not None else None,
            'buffer_exists': self.buffer is not None,
            'required_buffer_length': self.filter_length - 1,
            'filter_length': self.filter_length,
            'spatial_dims': self.spatial_dims,
            'n_features': self.n_features
        }
    
    def estimate_memory_usage(self, chunk_size: int, dtype: np.dtype = np.dtype(np.float64)) -> dict:
        """
        Estimate memory usage for design matrix construction.
        
        Parameters
        ----------
        chunk_size : int
            Size of stimulus chunks
        dtype : np.dtype, default=np.float64
            Data type for calculations
            
        Returns
        -------
        dict
            Memory usage estimates
        """
        # Memory for design matrix
        design_matrix_memory = chunk_size * self.n_features * dtype.itemsize
        
        # Memory for padded stimulus
        padded_length = chunk_size + self.filter_length - 1
        if self.spatial_dims is None:
            padded_memory = padded_length * dtype.itemsize
        else:
            padded_memory = padded_length * self.spatial_size * dtype.itemsize
        
        # Memory for buffer
        buffer_memory = (self.filter_length - 1) * self.spatial_size * dtype.itemsize
        
        total_memory = design_matrix_memory + padded_memory + buffer_memory
        
        return {
            'design_matrix_mb': design_matrix_memory / (1024**2),
            'padded_stimulus_mb': padded_memory / (1024**2),
            'buffer_mb': buffer_memory / (1024**2),
            'total_mb': total_memory / (1024**2),
            'chunk_size': chunk_size,
            'n_features': self.n_features,
            'dtype': str(dtype)
        }


def create_design_matrix_batch(stimulus: np.ndarray, filter_length: int,
                              spatial_dims: Optional[Tuple[int, ...]] = None,
                              n_colors: int = 1) -> np.ndarray:
    """
    Create design matrix for full stimulus array (non-streaming).
    
    This is a convenience function for small datasets that fit in memory.
    For large datasets, use StreamingDesignMatrix.
    
    Parameters
    ----------
    stimulus : np.ndarray
        Full stimulus array
    filter_length : int
        Number of time bins for filter
    spatial_dims : tuple, optional
        Spatial dimensions (height, width)
    n_colors : int, default=1
        Number of color channels
        
    Returns
    -------
    np.ndarray
        Design matrix with shape (n_time_bins, n_features)
    """
    # Create streaming builder
    builder = StreamingDesignMatrix(filter_length, spatial_dims, n_colors)
    
    # Build design matrix for full stimulus (as single chunk)
    design_matrix = builder.build_hankel_chunk(stimulus, is_first_chunk=True)
    
    return design_matrix


def validate_design_matrix(design_matrix: np.ndarray, stimulus_length: int,
                          filter_length: int, spatial_size: int = 1) -> dict:
    """
    Validate design matrix properties.
    
    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix to validate
    stimulus_length : int
        Length of original stimulus
    filter_length : int
        Filter length used
    spatial_size : int, default=1
        Size of spatial dimensions
        
    Returns
    -------
    dict
        Validation results
    """
    expected_n_time_bins = stimulus_length
    expected_n_features = filter_length * spatial_size
    
    validation_results = {
        'shape_ok': True,
        'finite_values': True,
        'expected_shape': (expected_n_time_bins, expected_n_features),
        'actual_shape': design_matrix.shape,
        'warnings': []
    }
    
    # Check shape
    if design_matrix.shape != (expected_n_time_bins, expected_n_features):
        validation_results['shape_ok'] = False
        validation_results['warnings'].append(
            f"Shape mismatch: expected {(expected_n_time_bins, expected_n_features)}, "
            f"got {design_matrix.shape}"
        )
    
    # Check for finite values
    if not np.isfinite(design_matrix).all():
        validation_results['finite_values'] = False
        n_nan = np.isnan(design_matrix).sum()
        n_inf = np.isinf(design_matrix).sum()
        validation_results['warnings'].append(
            f"Non-finite values found: {n_nan} NaN, {n_inf} infinite"
        )
    
    # Check for reasonable value range
    if np.isfinite(design_matrix).any():
        matrix_std = np.nanstd(design_matrix)
        matrix_max = np.nanmax(np.abs(design_matrix))
        
        if matrix_std == 0:
            validation_results['warnings'].append("Design matrix has zero variance")
        
        if matrix_max > 1e6:
            validation_results['warnings'].append(
                f"Design matrix has very large values (max: {matrix_max:.2e})"
            )
    
    return validation_results

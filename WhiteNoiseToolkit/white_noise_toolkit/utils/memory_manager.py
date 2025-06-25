"""
Memory management utilities for the White Noise Analysis Toolkit.

This module provides tools for monitoring and managing memory usage during
streaming analysis to prevent out-of-memory errors and optimize performance.
"""

import gc
import warnings
from typing import Tuple, Union, Optional
import numpy as np
import psutil
from ..core.exceptions import MemoryLimitError


class MemoryManager:
    """
    Monitor and manage memory usage during analysis.

    This class provides utilities for:
    - Monitoring current memory usage
    - Estimating optimal chunk sizes
    - Validating memory requirements
    - Adaptive memory management
    """

    def __init__(self, max_memory_gb: float = 8.0, warning_threshold: float = 0.8):
        """
        Initialize memory manager.

        Parameters
        ----------
        max_memory_gb : float, default=8.0
            Maximum memory usage in GB
        warning_threshold : float, default=0.8
            Fraction of max_memory_gb at which to issue warnings
        """
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.warning_threshold_bytes = self.max_memory_bytes * warning_threshold

        # Get system memory info
        self.system_memory = psutil.virtual_memory()

        # Validate that max_memory doesn't exceed system memory
        available_gb = self.system_memory.available / (1024**3)
        if max_memory_gb > available_gb * 0.9:  # Leave 10% buffer
            warnings.warn(
                f"Requested memory limit ({max_memory_gb:.1f} GB) is close to "
                f"available system memory ({available_gb:.1f} GB). "
                f"Consider reducing max_memory_gb.",
                UserWarning
            )

    def get_current_usage(self) -> float:
        """
        Get current memory usage in GB.

        Returns
        -------
        float
            Current memory usage in GB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024**3)

    def get_available_memory(self) -> float:
        """
        Get available memory in GB.

        Returns
        -------
        float
            Available memory in GB
        """
        current_usage = self.get_current_usage()
        return max(0, self.max_memory_gb - current_usage)

    def estimate_chunk_size(self, data_shape: Tuple[int, ...], dtype: Union[str, np.dtype],
                          n_chunks_target: int = 100, safety_factor: float = 0.5) -> int:
        """
        Estimate optimal chunk size based on available memory.

        Parameters
        ----------
        data_shape : tuple
            Shape of the full dataset
        dtype : str or np.dtype
            Data type
        n_chunks_target : int, default=100
            Target number of chunks
        safety_factor : float, default=0.5
            Safety factor to prevent memory overflow

        Returns
        -------
        int
            Recommended chunk size (number of time bins)
        """
        # Convert dtype to numpy dtype
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        # Calculate memory per sample
        if len(data_shape) == 1:
            # 1D temporal data
            memory_per_sample = dtype.itemsize
        else:
            # Multi-dimensional data
            sample_size = np.prod(data_shape[1:])  # All dimensions except time
            memory_per_sample = sample_size * dtype.itemsize

        # Get available memory
        available_memory = self.get_available_memory() * (1024**3)  # Convert to bytes
        available_memory *= safety_factor  # Apply safety factor

        # Calculate chunk size based on available memory
        chunk_size_memory = int(available_memory / memory_per_sample)

        # Calculate chunk size based on target number of chunks
        chunk_size_target = max(1, data_shape[0] // n_chunks_target)

        # Use the smaller of the two, but ensure minimum size
        chunk_size = min(chunk_size_memory, chunk_size_target)
        chunk_size = max(chunk_size, 100)  # Minimum chunk size

        # Don't exceed the total data size
        chunk_size = min(chunk_size, data_shape[0])

        return chunk_size

    def validate_memory_requirements(self, data_shape: Tuple[int, ...],
                                   dtype: Union[str, np.dtype],
                                   filter_length: int) -> None:
        """
        Check if analysis is feasible with current memory limits.

        Parameters
        ----------
        data_shape : tuple
            Shape of the dataset
        dtype : str or np.dtype
            Data type
        filter_length : int
            Filter length in time bins

        Raises
        ------
        MemoryLimitError
            If memory requirements exceed limits
        """
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        # Estimate memory requirements for design matrix
        n_time_bins = data_shape[0]
        if len(data_shape) == 1:
            n_features = filter_length
        else:
            spatial_size = np.prod(data_shape[1:])
            n_features = filter_length * spatial_size

        # Memory for design matrix (worst case: full matrix in memory)
        design_matrix_memory = n_time_bins * n_features * dtype.itemsize

        # Memory for accumulation matrices (for whitened STA)
        xtx_memory = n_features * n_features * dtype.itemsize
        xty_memory = n_features * dtype.itemsize

        # Total memory requirement
        total_memory = design_matrix_memory + xtx_memory + xty_memory
        total_memory_gb = total_memory / (1024**3)

        if total_memory_gb > self.max_memory_gb:
            raise MemoryLimitError(
                f"Analysis requires approximately {total_memory_gb:.1f} GB of memory",
                required_gb=total_memory_gb,
                available_gb=self.max_memory_gb
            )

    def adaptive_chunk_size(self, base_chunk_size: int, current_usage: Optional[float] = None) -> int:
        """
        Dynamically adjust chunk size based on memory usage.

        Parameters
        ----------
        base_chunk_size : int
            Base chunk size
        current_usage : float, optional
            Current memory usage in GB. If None, will be measured.

        Returns
        -------
        int
            Adjusted chunk size
        """
        if current_usage is None:
            current_usage = self.get_current_usage()

        # Calculate memory pressure
        memory_pressure = current_usage / self.max_memory_gb

        if memory_pressure > 0.9:
            # High memory pressure - reduce chunk size significantly
            factor = 0.5
        elif memory_pressure > self.warning_threshold:
            # Moderate memory pressure - reduce chunk size
            factor = 0.7
        elif memory_pressure < 0.3:
            # Low memory pressure - can increase chunk size
            factor = 1.5
        else:
            # Normal memory pressure - keep current size
            factor = 1.0

        new_chunk_size = int(base_chunk_size * factor)

        # Ensure minimum and maximum bounds
        new_chunk_size = max(new_chunk_size, 50)  # Minimum chunk size
        new_chunk_size = min(new_chunk_size, 10000)  # Maximum chunk size

        return new_chunk_size

    def check_memory_warning(self) -> None:
        """
        Check memory usage and issue warning if necessary.
        """
        current_usage = self.get_current_usage()

        if current_usage > self.max_memory_gb * self.warning_threshold:
            warnings.warn(
                f"Memory usage ({current_usage:.1f} GB) is approaching limit "
                f"({self.max_memory_gb:.1f} GB). Consider reducing chunk size "
                f"or increasing memory limit.",
                UserWarning
            )

    def cleanup_large_arrays(self, *arrays) -> None:
        """
        Explicitly delete large arrays and trigger garbage collection.

        Parameters
        ----------
        *arrays
            Arrays to delete
        """
        for arr in arrays:
            if hasattr(arr, '__del__'):
                del arr
        gc.collect()

    def get_memory_info(self) -> dict:
        """
        Get comprehensive memory information.

        Returns
        -------
        dict
            Dictionary with memory statistics
        """
        current_usage = self.get_current_usage()
        available = self.get_available_memory()
        system_info = psutil.virtual_memory()

        return {
            'current_usage_gb': current_usage,
            'max_limit_gb': self.max_memory_gb,
            'available_gb': available,
            'usage_fraction': current_usage / self.max_memory_gb,
            'system_total_gb': system_info.total / (1024**3),
            'system_available_gb': system_info.available / (1024**3),
            'system_used_gb': system_info.used / (1024**3),
            'system_percent': system_info.percent
        }

    def estimate_processing_time(self, data_shape: Tuple[int, ...],
                               chunk_size: int) -> dict:
        """
        Estimate processing time based on data size and chunk size.

        Parameters
        ----------
        data_shape : tuple
            Shape of dataset
        chunk_size : int
            Chunk size for processing

        Returns
        -------
        dict
            Time estimates
        """
        n_time_bins = data_shape[0]
        n_chunks = int(np.ceil(n_time_bins / chunk_size))

        # Rough estimates based on typical performance
        time_per_chunk = 0.1  # seconds per chunk (rough estimate)
        total_time = n_chunks * time_per_chunk

        return {
            'n_chunks': n_chunks,
            'estimated_time_seconds': total_time,
            'estimated_time_minutes': total_time / 60,
            'time_per_chunk_seconds': time_per_chunk,
            'chunk_size': chunk_size
        }

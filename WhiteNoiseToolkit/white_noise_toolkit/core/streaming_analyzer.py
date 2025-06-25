"""
Streaming data generators for the White Noise Analysis Toolkit.

This module provides generator functions for streaming large datasets
in chunks, with support for memory mapping and proper temporal alignment.
"""

import warnings
from typing import Generator, Union, Tuple, Optional, Any
import numpy as np
import h5py
from pathlib import Path
from ..core.exceptions import DataValidationError, ConfigurationError


def create_stimulus_generator(
    stimulus_data: Union[np.ndarray, str, Path], 
    chunk_size: int = 1000,
    memory_mapped: bool = True
) -> Generator[np.ndarray, None, None]:
    """
    Create generator that yields stimulus chunks of specified size.
    
    Parameters
    ----------
    stimulus_data : np.ndarray or str or Path
        Stimulus array or path to HDF5 file
    chunk_size : int, default=1000
        Size of chunks to yield
    memory_mapped : bool, default=True
        Use memory mapping for large files
        
    Yields
    ------
    np.ndarray
        Stimulus chunk with shape (chunk_size, *spatial_dims)
        
    Raises
    ------
    DataValidationError
        If stimulus data is invalid
    """
    # Handle different input types
    if isinstance(stimulus_data, (str, Path)):
        # Load from file
        file_path = Path(stimulus_data)
        
        if file_path.suffix.lower() in ['.h5', '.hdf5']:
            # HDF5 file
            with h5py.File(file_path, 'r') as f:
                # Try common key names
                stimulus_keys = ['stimulus', 'stim', 'data']
                stimulus_key = None
                
                for key in stimulus_keys:
                    if key in f:
                        stimulus_key = key
                        break
                
                if stimulus_key is None:
                    available_keys = list(f.keys())
                    raise DataValidationError(
                        f"Could not find stimulus data in HDF5 file. "
                        f"Available keys: {available_keys}. "
                        f"Expected one of: {stimulus_keys}"
                    )
                
                stimulus_dataset = f[stimulus_key]
                
                # Validate dataset - ensure it's actually a dataset, not a group
                if not isinstance(stimulus_dataset, h5py.Dataset):
                    raise DataValidationError(
                        f"'{stimulus_key}' is not a dataset in HDF5 file"
                    )
                
                if len(stimulus_dataset.shape) == 0:
                    raise DataValidationError("Stimulus dataset is empty")
                
                n_time_bins = stimulus_dataset.shape[0]
                
                # Yield chunks
                for start_idx in range(0, n_time_bins, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_time_bins)
                    chunk = stimulus_dataset[start_idx:end_idx]
                    yield np.array(chunk)
                    
                return  # Exit after processing HDF5 file
        
        elif file_path.suffix.lower() in ['.npy']:
            # NumPy file
            if memory_mapped:
                stimulus_data = np.load(file_path, mmap_mode='r')
            else:
                stimulus_data = np.load(file_path)
        
        else:
            raise DataValidationError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: .h5, .hdf5, .npy"
            )
    
    # Handle numpy array (including memory-mapped)
    if isinstance(stimulus_data, np.ndarray):
        # Validate stimulus data
        if stimulus_data.size == 0:
            raise DataValidationError("Stimulus data is empty")
        
        if len(stimulus_data.shape) == 0:
            raise DataValidationError("Stimulus data has no dimensions")
        
        n_time_bins = stimulus_data.shape[0]
        
        # Yield chunks
        for start_idx in range(0, n_time_bins, chunk_size):
            end_idx = min(start_idx + chunk_size, n_time_bins)
            chunk = stimulus_data[start_idx:end_idx]
            
            # Ensure chunk is contiguous
            if not chunk.flags.c_contiguous:
                chunk = np.ascontiguousarray(chunk)
            
            yield chunk
    
    else:
        raise DataValidationError(
            f"Invalid stimulus data type: {type(stimulus_data)}. "
            f"Expected np.ndarray, str, or Path."
        )


def create_spike_generator(
    spike_data: Union[np.ndarray, str, Path],
    chunk_size: int = 1000,
    align_with_stimulus: bool = True
) -> Generator[np.ndarray, None, None]:
    """
    Create generator that yields spike chunks aligned with stimulus chunks.
    
    Parameters
    ----------
    spike_data : np.ndarray or str or Path
        Spike array or path to HDF5 file. Should contain binned spike counts.
    chunk_size : int, default=1000
        Size of chunks to yield (must match stimulus chunks)
    align_with_stimulus : bool, default=True
        Ensure temporal alignment across chunks
        
    Yields
    ------
    np.ndarray
        Spike chunk with shape (chunk_size,) or (chunk_size, n_electrodes)
        
    Raises
    ------
    DataValidationError
        If spike data is invalid
    """
    # Handle different input types
    if isinstance(spike_data, (str, Path)):
        # Load from file
        file_path = Path(spike_data)
        
        if file_path.suffix.lower() in ['.h5', '.hdf5']:
            # HDF5 file
            with h5py.File(file_path, 'r') as f:
                # Try common key names
                spike_keys = ['spikes', 'spike_counts', 'response', 'data']
                spike_key = None
                
                for key in spike_keys:
                    if key in f:
                        spike_key = key
                        break
                
                if spike_key is None:
                    available_keys = list(f.keys())
                    raise DataValidationError(
                        f"Could not find spike data in HDF5 file. "
                        f"Available keys: {available_keys}. "
                        f"Expected one of: {spike_keys}"
                    )
                
                spike_dataset = f[spike_key]
                
                # Validate dataset - ensure it's actually a dataset, not a group
                if not isinstance(spike_dataset, h5py.Dataset):
                    raise DataValidationError(
                        f"'{spike_key}' is not a dataset in HDF5 file"
                    )
                
                if len(spike_dataset.shape) == 0:
                    raise DataValidationError("Spike dataset is empty")
                
                n_time_bins = spike_dataset.shape[0]
                
                # Yield chunks
                for start_idx in range(0, n_time_bins, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_time_bins)
                    chunk = spike_dataset[start_idx:end_idx]
                    yield np.array(chunk)
                    
                return  # Exit after processing HDF5 file
        
        elif file_path.suffix.lower() in ['.npy']:
            # NumPy file
            spike_data = np.load(file_path)
        
        else:
            raise DataValidationError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: .h5, .hdf5, .npy"
            )
    
    # Handle numpy array
    if isinstance(spike_data, np.ndarray):
        # Validate spike data
        if spike_data.size == 0:
            raise DataValidationError("Spike data is empty")
        
        if len(spike_data.shape) == 0:
            raise DataValidationError("Spike data has no dimensions")
        
        # Check for non-negative values (spike counts should be >= 0)
        if np.any(spike_data < 0):
            warnings.warn("Spike data contains negative values. This is unusual for spike counts.")
        
        n_time_bins = spike_data.shape[0]
        
        # Yield chunks
        for start_idx in range(0, n_time_bins, chunk_size):
            end_idx = min(start_idx + chunk_size, n_time_bins)
            chunk = spike_data[start_idx:end_idx]
            
            # Ensure chunk is contiguous
            if not chunk.flags.c_contiguous:
                chunk = np.ascontiguousarray(chunk)
            
            yield chunk
    
    else:
        raise DataValidationError(
            f"Invalid spike data type: {type(spike_data)}. "
            f"Expected np.ndarray, str, or Path."
        )


def validate_generators(
    stimulus_gen: Generator[np.ndarray, None, None],
    spike_gen: Generator[np.ndarray, None, None],
    max_validation_chunks: int = 3
) -> dict:
    """
    Validate that generators are properly aligned and configured.
    
    Parameters
    ----------
    stimulus_gen : Generator
        Stimulus generator
    spike_gen : Generator
        Spike generator
    max_validation_chunks : int, default=3
        Maximum number of chunks to validate
        
    Returns
    -------
    dict
        Validation results
        
    Raises
    ------
    DataValidationError
        If generators are misaligned or invalid
    """
    validation_results = {
        'chunks_validated': 0,
        'stimulus_shapes': [],
        'spike_shapes': [],
        'alignment_ok': True,
        'warnings': []
    }
    
    try:
        # Validate first few chunks
        for i in range(max_validation_chunks):
            try:
                stim_chunk = next(stimulus_gen)
                spike_chunk = next(spike_gen)
            except StopIteration:
                # One generator ended before the other
                if i == 0:
                    raise DataValidationError("Generators are empty")
                else:
                    validation_results['warnings'].append(
                        f"Generators have different lengths (validated {i} chunks)"
                    )
                    break
            
            validation_results['chunks_validated'] += 1
            validation_results['stimulus_shapes'].append(stim_chunk.shape)
            validation_results['spike_shapes'].append(spike_chunk.shape)
            
            # Check temporal alignment
            if stim_chunk.shape[0] != spike_chunk.shape[0]:
                validation_results['alignment_ok'] = False
                raise DataValidationError(
                    f"Chunk {i}: Temporal dimensions don't match. "
                    f"Stimulus: {stim_chunk.shape[0]}, Spikes: {spike_chunk.shape[0]}"
                )
            
            # Check for NaN or infinite values
            if not np.isfinite(stim_chunk).all():
                validation_results['warnings'].append(
                    f"Chunk {i}: Stimulus contains NaN or infinite values"
                )
            
            if not np.isfinite(spike_chunk).all():
                validation_results['warnings'].append(
                    f"Chunk {i}: Spike data contains NaN or infinite values"
                )
            
            # Check spike data range
            if np.any(spike_chunk < 0):
                validation_results['warnings'].append(
                    f"Chunk {i}: Spike data contains negative values"
                )
    
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        else:
            raise DataValidationError(f"Generator validation failed: {e}")
    
    return validation_results


def create_aligned_generators(
    stimulus_data: Union[np.ndarray, str, Path],
    spike_data: Union[np.ndarray, str, Path],
    chunk_size: int = 1000,
    validate: bool = True
) -> Tuple[Generator[np.ndarray, None, None], Generator[np.ndarray, None, None]]:
    """
    Create aligned stimulus and spike generators.
    
    Parameters
    ----------
    stimulus_data : np.ndarray or str or Path
        Stimulus data or path to file
    spike_data : np.ndarray or str or Path
        Spike data or path to file
    chunk_size : int, default=1000
        Chunk size for streaming
    validate : bool, default=True
        Whether to validate generator alignment
        
    Returns
    -------
    tuple
        (stimulus_generator, spike_generator)
        
    Raises
    ------
    DataValidationError
        If data cannot be loaded or generators are misaligned
    """
    # Create generators
    stimulus_gen = create_stimulus_generator(stimulus_data, chunk_size)
    spike_gen = create_spike_generator(spike_data, chunk_size)
    
    # Validate if requested
    if validate:
        # Create new generators for validation (since generators are consumed)
        validation_stim_gen = create_stimulus_generator(stimulus_data, chunk_size)
        validation_spike_gen = create_spike_generator(spike_data, chunk_size)
        
        validation_results = validate_generators(
            validation_stim_gen, validation_spike_gen
        )
        
        # Log warnings
        if validation_results['warnings']:
            warnings.warn(
                f"Generator validation warnings: {validation_results['warnings']}"
            )
        
        if not validation_results['alignment_ok']:
            raise DataValidationError("Generator alignment validation failed")
    
    return stimulus_gen, spike_gen


def estimate_total_chunks(
    data: Union[np.ndarray, str, Path],
    chunk_size: int
) -> int:
    """
    Estimate total number of chunks for progress tracking.
    
    Parameters
    ----------
    data : np.ndarray or str or Path
        Data array or path to file
    chunk_size : int
        Chunk size
        
    Returns
    -------
    int
        Estimated number of chunks
    """
    if isinstance(data, np.ndarray):
        n_time_bins = data.shape[0]
    elif isinstance(data, (str, Path)):
        file_path = Path(data)
        
        if file_path.suffix.lower() in ['.h5', '.hdf5']:
            with h5py.File(file_path, 'r') as f:
                # Try to find the main dataset
                keys = ['stimulus', 'stim', 'spikes', 'spike_counts', 'data']
                dataset_key = None
                
                for key in keys:
                    if key in f:
                        dataset_key = key
                        break
                
                if dataset_key is None and len(f.keys()) > 0:
                    dataset_key = list(f.keys())[0]
                
                if dataset_key is not None:
                    dataset = f[dataset_key]
                    if isinstance(dataset, h5py.Dataset):
                        n_time_bins = dataset.shape[0]
                    else:
                        n_time_bins = 1000  # Fallback
                else:
                    n_time_bins = 0
        
        elif file_path.suffix.lower() in ['.npy']:
            # Get shape without loading full array
            with open(file_path, 'rb') as f:
                # Read numpy header to get shape
                version = np.lib.format.read_magic(f)
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                n_time_bins = shape[0] if len(shape) > 0 else 0
        
        else:
            # Unknown format, return conservative estimate
            n_time_bins = 1000
    
    else:
        n_time_bins = 1000  # Default fallback
    
    return int(np.ceil(n_time_bins / chunk_size))


def chunk_data_info(
    stimulus_data: Union[np.ndarray, str, Path],
    spike_data: Union[np.ndarray, str, Path],
    chunk_size: int
) -> dict:
    """
    Get information about data chunking for memory and performance planning.
    
    Parameters
    ----------
    stimulus_data : np.ndarray or str or Path
        Stimulus data
    spike_data : np.ndarray or str or Path
        Spike data
    chunk_size : int
        Chunk size
        
    Returns
    -------
    dict
        Information about chunking
    """
    stim_chunks = estimate_total_chunks(stimulus_data, chunk_size)
    spike_chunks = estimate_total_chunks(spike_data, chunk_size)
    
    # Estimate memory per chunk
    if isinstance(stimulus_data, np.ndarray):
        stim_shape = stimulus_data.shape
        stim_dtype = stimulus_data.dtype
    else:
        # Conservative estimates
        stim_shape = (chunk_size,)
        stim_dtype = np.float64
    
    if isinstance(spike_data, np.ndarray):
        spike_shape = spike_data.shape
        spike_dtype = spike_data.dtype
    else:
        # Conservative estimates
        spike_shape = (chunk_size,)
        spike_dtype = np.float64
    
    # Memory per chunk (rough estimate)
    stim_memory_per_chunk = chunk_size * np.prod(stim_shape[1:]) * stim_dtype.itemsize
    spike_memory_per_chunk = chunk_size * np.prod(spike_shape[1:]) * spike_dtype.itemsize
    total_memory_per_chunk = stim_memory_per_chunk + spike_memory_per_chunk
    
    return {
        'stimulus_chunks': stim_chunks,
        'spike_chunks': spike_chunks,
        'total_chunks': max(stim_chunks, spike_chunks),
        'chunk_size': chunk_size,
        'estimated_memory_per_chunk_mb': total_memory_per_chunk / (1024**2),
        'estimated_total_memory_mb': (stim_chunks * total_memory_per_chunk) / (1024**2),
        'stimulus_shape_estimate': stim_shape,
        'spike_shape_estimate': spike_shape
    }

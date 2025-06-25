"""
I/O handling utilities for white noise analysis.

This module provides utilities for reading and writing various data formats
commonly used in neurophysiology experiments.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import h5py
from scipy.io import loadmat, savemat
import pickle
import json
import yaml

from ..core.exceptions import DataValidationError, FileFormatError
from .logging_config import get_logger

logger = get_logger(__name__)


class DataReader:
    """
    Generic data reader supporting multiple formats.

    Supports:
    - HDF5 (.h5, .hdf5)
    - MAT files (.mat)
    - NumPy files (.npy, .npz)
    - Pickle files (.pkl)
    - JSON (.json)
    - YAML (.yaml, .yml)
    """

    @staticmethod
    def read(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Read data from file, automatically detecting format.

        Parameters
        ----------
        filepath : str or Path
            Path to the data file

        Returns
        -------
        dict
            Dictionary containing the loaded data

        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        FileFormatError
            If file format is not supported or corrupted
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        try:
            if suffix in ['.h5', '.hdf5']:
                return DataReader._read_hdf5(filepath)
            elif suffix == '.mat':
                return DataReader._read_mat(filepath)
            elif suffix == '.npy':
                return {'data': np.load(filepath)}
            elif suffix == '.npz':
                return dict(np.load(filepath))
            elif suffix == '.pkl':
                return DataReader._read_pickle(filepath)
            elif suffix == '.json':
                return DataReader._read_json(filepath)
            elif suffix in ['.yaml', '.yml']:
                return DataReader._read_yaml(filepath)
            else:
                raise FileFormatError(f"Unsupported file format: {suffix}")

        except Exception as e:
            if isinstance(e, (FileNotFoundError, FileFormatError)):
                raise
            raise FileFormatError(f"Error reading {filepath}: {str(e)}")

    @staticmethod
    def _read_hdf5(filepath: Path) -> Dict[str, Any]:
        """Read HDF5 file."""
        data = {}
        with h5py.File(filepath, 'r') as f:
            def _extract_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
                elif isinstance(obj, h5py.Group):
                    # Handle groups recursively
                    group_data = {}
                    obj.visititems(lambda n, o:
                        group_data.update({n: o[()]}) if isinstance(o, h5py.Dataset) else None)
                    if group_data:
                        data[name] = group_data

            f.visititems(_extract_item)
        return data

    @staticmethod
    def _read_mat(filepath: Path) -> Dict[str, Any]:
        """Read MATLAB file."""
        try:
            # Try scipy.io.loadmat first
            data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
            # Remove MATLAB metadata
            return {k: v for k, v in data.items() if not k.startswith('__')}
        except Exception:
            # Fallback for newer MATLAB files
            try:
                import h5py
                return DataReader._read_hdf5(filepath)
            except Exception as e:
                raise FileFormatError(f"Cannot read MATLAB file {filepath}: {str(e)}")

    @staticmethod
    def _read_pickle(filepath: Path) -> Dict[str, Any]:
        """Read pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            data = {'data': data}
        return data

    @staticmethod
    def _read_json(filepath: Path) -> Dict[str, Any]:
        """Read JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def _read_yaml(filepath: Path) -> Dict[str, Any]:
        """Read YAML file."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)


class DataWriter:
    """
    Generic data writer supporting multiple formats.
    """

    @staticmethod
    def write(data: Dict[str, Any], filepath: Union[str, Path],
              format: Optional[str] = None, **kwargs) -> None:
        """
        Write data to file.

        Parameters
        ----------
        data : dict
            Data to write
        filepath : str or Path
            Output file path
        format : str, optional
            Output format. If None, inferred from file extension
        **kwargs
            Additional arguments passed to the writer
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = filepath.suffix.lower()

        try:
            if format in ['.h5', '.hdf5']:
                DataWriter._write_hdf5(data, filepath, **kwargs)
            elif format == '.mat':
                DataWriter._write_mat(data, filepath, **kwargs)
            elif format == '.npz':
                DataWriter._write_npz(data, filepath, **kwargs)
            elif format == '.pkl':
                DataWriter._write_pickle(data, filepath, **kwargs)
            elif format == '.json':
                DataWriter._write_json(data, filepath, **kwargs)
            elif format in ['.yaml', '.yml']:
                DataWriter._write_yaml(data, filepath, **kwargs)
            else:
                raise FileFormatError(f"Unsupported output format: {format}")

        except Exception as e:
            if isinstance(e, FileFormatError):
                raise
            raise FileFormatError(f"Error writing to {filepath}: {str(e)}")

    @staticmethod
    def _write_hdf5(data: Dict[str, Any], filepath: Path, **kwargs) -> None:
        """Write HDF5 file."""
        compression = kwargs.get('compression', 'gzip')

        with h5py.File(filepath, 'w') as f:
            def _write_item(group, key, value):
                if isinstance(value, dict):
                    subgroup = group.create_group(key)
                    for subkey, subvalue in value.items():
                        _write_item(subgroup, subkey, subvalue)
                else:
                    try:
                        if isinstance(value, (list, tuple)):
                            value = np.array(value)
                        group.create_dataset(key, data=value, compression=compression)
                    except Exception as e:
                        logger.warning(f"Could not write {key}: {str(e)}")

            for key, value in data.items():
                _write_item(f, key, value)

    @staticmethod
    def _write_mat(data: Dict[str, Any], filepath: Path, **kwargs) -> None:
        """Write MATLAB file."""
        # Convert numpy arrays to appropriate format
        mat_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                mat_data[key] = value
            elif isinstance(value, dict):
                # Flatten nested dictionaries with dot notation
                for subkey, subvalue in value.items():
                    mat_data[f"{key}_{subkey}"] = subvalue
            else:
                mat_data[key] = value

        savemat(filepath, mat_data, **kwargs)

    @staticmethod
    def _write_npz(data: Dict[str, Any], filepath: Path, **kwargs) -> None:
        """Write NumPy compressed file."""
        # Flatten nested dictionaries
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_data[f"{key}_{subkey}"] = subvalue
            else:
                flat_data[key] = value

        np.savez_compressed(filepath, **flat_data)

    @staticmethod
    def _write_pickle(data: Dict[str, Any], filepath: Path, **kwargs) -> None:
        """Write pickle file."""
        protocol = kwargs.get('protocol', pickle.HIGHEST_PROTOCOL)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)

    @staticmethod
    def _write_json(data: Dict[str, Any], filepath: Path, **kwargs) -> None:
        """Write JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def _convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        json_data = _convert_numpy(data)

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=kwargs.get('indent', 2))

    @staticmethod
    def _write_yaml(data: Dict[str, Any], filepath: Path, **kwargs) -> None:
        """Write YAML file."""
        # Convert numpy arrays to lists for YAML serialization
        def _convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        yaml_data = _convert_numpy(data)

        with open(filepath, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, **kwargs)


class SpikeDataLoader:
    """
    Specialized loader for spike data formats.

    Supports common spike data formats and provides validation.
    """

    @staticmethod
    def load_spike_times(filepath: Union[str, Path],
                        format: Optional[str] = None,
                        unit_id: Optional[int] = None,
                        time_column: str = 'spike_times',
                        unit_column: str = 'unit_id') -> np.ndarray:
        """
        Load spike times from file.

        Parameters
        ----------
        filepath : str or Path
            Path to spike data file
        format : str, optional
            File format override
        unit_id : int, optional
            Specific unit ID to load
        time_column : str
            Column name for spike times
        unit_column : str
            Column name for unit IDs

        Returns
        -------
        numpy.ndarray
            Array of spike times
        """
        data = DataReader.read(filepath)

        # Handle different data structures
        if time_column in data:
            spike_times = np.array(data[time_column])

            if unit_id is not None and unit_column in data:
                unit_ids = np.array(data[unit_column])
                mask = unit_ids == unit_id
                spike_times = spike_times[mask]

        elif 'data' in data:
            spike_times = np.array(data['data'])

        else:
            # Try to find spike times in the data
            for key, value in data.items():
                if isinstance(value, np.ndarray) and 'spike' in key.lower():
                    spike_times = value
                    break
            else:
                raise DataValidationError("Could not find spike times in data")

        # Validate spike times
        spike_times = np.sort(spike_times.flatten())

        if len(spike_times) == 0:
            warnings.warn("No spike times found")

        if np.any(np.diff(spike_times) < 0):
            warnings.warn("Spike times are not sorted")
            spike_times = np.sort(spike_times)

        logger.info(f"Loaded {len(spike_times)} spike times")
        return spike_times

    @staticmethod
    def validate_spike_data(spike_times: np.ndarray,
                          duration: Optional[float] = None) -> None:
        """
        Validate spike time data.

        Parameters
        ----------
        spike_times : numpy.ndarray
            Array of spike times
        duration : float, optional
            Expected experiment duration

        Raises
        ------
        DataValidationError
            If data is invalid
        """
        if not isinstance(spike_times, np.ndarray):
            raise DataValidationError("Spike times must be numpy array")

        if spike_times.ndim != 1:
            raise DataValidationError("Spike times must be 1D array")

        if len(spike_times) == 0:
            warnings.warn("Empty spike times array")
            return

        if np.any(spike_times < 0):
            raise DataValidationError("Spike times cannot be negative")

        if not np.all(np.diff(spike_times) >= 0):
            raise DataValidationError("Spike times must be sorted")

        if duration is not None and np.any(spike_times > duration):
            raise DataValidationError(f"Spike times exceed duration {duration}")

        # Check for reasonable ISI values
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            min_isi = np.min(isis)

            if min_isi < 1e-6:  # 1 microsecond
                warnings.warn(f"Very short ISI detected: {min_isi} seconds")


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a data file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file

    Returns
    -------
    dict
        File information including size, format, and basic content info
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    info = {
        'path': str(filepath),
        'name': filepath.name,
        'size_bytes': filepath.stat().st_size,
        'size_mb': filepath.stat().st_size / (1024 * 1024),
        'suffix': filepath.suffix.lower(),
        'modified': filepath.stat().st_mtime
    }

    # Try to get content info
    try:
        data = DataReader.read(filepath)
        info['keys'] = list(data.keys())
        info['num_keys'] = len(data)

        # Get info about arrays
        array_info = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                array_info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'size': value.size
                }

        if array_info:
            info['arrays'] = array_info

    except Exception as e:
        info['error'] = str(e)

    return info


# Convenience functions
def load_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function for loading data."""
    return DataReader.read(filepath)


def save_data(data: Dict[str, Any], filepath: Union[str, Path], **kwargs) -> None:
    """Convenience function for saving data."""
    DataWriter.write(data, filepath, **kwargs)


def load_spikes(filepath: Union[str, Path], **kwargs) -> np.ndarray:
    """Convenience function for loading spike times."""
    return SpikeDataLoader.load_spike_times(filepath, **kwargs)

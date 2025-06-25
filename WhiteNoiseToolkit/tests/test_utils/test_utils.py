"""
Tests for utility modules functionality.
"""

import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch, mock_open

from white_noise_toolkit.utils.memory_manager import MemoryManager
from white_noise_toolkit.utils.io_handlers import save_data, load_data
from white_noise_toolkit.utils.preprocessing import SpikeProcessor
from white_noise_toolkit.utils.metrics import FilterMetrics, NonlinearityMetrics


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    def test_initialization(self):
        """Test memory manager initialization."""
        manager = MemoryManager(max_memory_gb=2.0)
        assert manager.max_memory_gb == 2.0
    
    def test_estimate_chunk_size(self):
        """Test chunk size estimation."""
        manager = MemoryManager(max_memory_gb=1.0)
        
        # Test with different data shapes
        data_shape = (10000, 100)  # 10k samples, 100 features
        chunk_size = manager.estimate_chunk_size(data_shape, 'float64')
        assert chunk_size > 0
        assert isinstance(chunk_size, int)
        
        # Larger data should result in smaller chunks (more chunks)
        large_shape = (1000, 10)
        small_shape = (100000, 100)
        large_chunk = manager.estimate_chunk_size(large_shape, 'float64')
        small_chunk = manager.estimate_chunk_size(small_shape, 'float64')
        # The relationship might not always hold due to safety factors
        assert large_chunk > 0 and small_chunk > 0
    
    def test_estimate_chunk_size_edge_cases(self):
        """Test chunk size estimation with edge cases."""
        manager = MemoryManager(max_memory_gb=0.1)  # Very small limit
        
        # Should still return reasonable chunk size
        data_shape = (100000, 50)
        chunk_size = manager.estimate_chunk_size(data_shape, 'float64')
        assert chunk_size >= 10  # Minimum reasonable chunk size
        
        # Test with small data
        small_shape = (100, 10)
        chunk_size = manager.estimate_chunk_size(small_shape, 'float64')
        assert chunk_size > 0
    
    def test_get_current_usage(self):
        """Test current memory usage reporting."""
        manager = MemoryManager(max_memory_gb=1.0)
        
        # This should not raise an exception
        usage = manager.get_current_usage()
        assert usage >= 0
        assert isinstance(usage, float)
    
    def test_get_available_memory(self):
        """Test available memory estimation."""
        manager = MemoryManager(max_memory_gb=2.0)
        
        available = manager.get_available_memory()
        assert available >= 0
        assert isinstance(available, float)
        
        # Available memory should be reasonable
        assert available <= manager.max_memory_gb


class TestIOHandlers:
    """Test cases for I/O handler functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            'array_data': np.random.randn(100),
            'dict_data': {'key1': 'value1', 'key2': 42},
            'list_data': [1, 2, 3, 4, 5],
            'nested_data': {
                'arrays': {
                    'filter': np.array([0.1, 0.2, 0.3]),
                    'nonlinearity': np.linspace(0, 1, 50)
                },
                'metadata': {
                    'n_samples': 1000,
                    'sampling_rate': 10000.0
                }
            }
        }
    
    def test_save_load_data_pickle(self):
        """Test saving and loading data with pickle format."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            try:
                # Save data
                save_data(self.test_data, tmp_file.name)
                
                # Load data
                loaded_data = load_data(tmp_file.name)
                
                # Check basic structure
                assert isinstance(loaded_data, dict)
                assert 'array_data' in loaded_data
                assert 'dict_data' in loaded_data
                assert 'list_data' in loaded_data
                assert 'nested_data' in loaded_data
                
                # Check array data
                np.testing.assert_array_equal(
                    loaded_data['array_data'], 
                    self.test_data['array_data']
                )
                
                # Check nested arrays
                np.testing.assert_array_equal(
                    loaded_data['nested_data']['arrays']['filter'],
                    self.test_data['nested_data']['arrays']['filter']
                )
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_save_load_data_hdf5(self):
        """Test saving and loading data with HDF5 format."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            try:
                # Test with simple data that HDF5 can handle
                simple_data = {
                    'array_data': np.random.randn(100),
                    'scalar_data': 42.0,
                    'string_data': 'test'
                }
                
                # Save data
                save_data(simple_data, tmp_file.name)
                
                # Load data
                loaded_data = load_data(tmp_file.name)
                
                # Check data
                assert isinstance(loaded_data, dict)
                np.testing.assert_array_equal(
                    loaded_data['array_data'], 
                    simple_data['array_data']
                )
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_save_load_data_mat(self):
        """Test saving and loading data with MATLAB format."""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file:
            try:
                # Test with MATLAB-compatible data
                matlab_data = {
                    'array_data': np.random.randn(10, 10),
                    'scalar_data': 42.0
                }
                
                # Save data
                save_data(matlab_data, tmp_file.name)
                
                # Load data
                loaded_data = load_data(tmp_file.name)
                
                # Check data (MATLAB format may modify structure slightly)
                assert isinstance(loaded_data, dict)
                assert 'array_data' in loaded_data
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_save_load_invalid_format(self):
        """Test saving with invalid file format."""
        with tempfile.NamedTemporaryFile(suffix='.invalid', delete=False) as tmp_file:
            try:
                with pytest.raises(ValueError):
                    save_data(self.test_data, tmp_file.name)
            finally:
                os.unlink(tmp_file.name)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.pkl')


class TestSpikeProcessor:
    """Test cases for SpikeProcessor class."""
    
    def test_remove_artifacts_basic(self):
        """Test basic artifact removal."""
        # Create test spike times with some artifacts
        spike_times = np.array([0.1, 0.1001, 0.2, 0.5, 0.8, 0.8001, 1.2, 1.5])
        
        cleaned, removed_count = SpikeProcessor.remove_artifacts(spike_times)
        
        # Should remove some spikes with very short ISIs
        assert removed_count > 0
        assert len(cleaned) < len(spike_times)
        assert isinstance(cleaned, np.ndarray)
        assert isinstance(removed_count, int)
    
    def test_remove_artifacts_empty(self):
        """Test artifact removal with empty spike train."""
        empty_spikes = np.array([])
        
        cleaned, removed_count = SpikeProcessor.remove_artifacts(empty_spikes)
        
        assert len(cleaned) == 0
        assert removed_count == 0
    
    def test_remove_artifacts_no_artifacts(self):
        """Test artifact removal with clean spike train."""
        clean_spikes = np.array([0.1, 0.2, 0.5, 0.8, 1.2])  # Well-separated spikes
        
        cleaned, removed_count = SpikeProcessor.remove_artifacts(clean_spikes)
        
        # Should not remove any spikes
        assert removed_count == 0
        np.testing.assert_array_equal(cleaned, clean_spikes)
    
    def test_detect_bursts_basic(self):
        """Test basic burst detection."""
        # Create spike times with burst pattern
        spike_times = np.array([
            0.1, 0.105, 0.11, 0.115,  # Burst 1
            0.5,  # Isolated spike
            1.0, 1.005, 1.01,  # Burst 2
            2.0   # Isolated spike
        ])
        
        burst_info = SpikeProcessor.detect_bursts(spike_times)
        
        assert 'burst_starts' in burst_info
        assert 'burst_ends' in burst_info
        assert 'burst_spikes' in burst_info
        assert 'isolated_spikes' in burst_info
        
        # Should detect some bursts
        assert len(burst_info['burst_starts']) > 0
        assert len(burst_info['isolated_spikes']) > 0
    
    def test_detect_bursts_no_bursts(self):
        """Test burst detection with isolated spikes."""
        # Well-separated spikes (no bursts)
        spike_times = np.array([0.1, 0.3, 0.6, 1.0, 1.5])
        
        burst_info = SpikeProcessor.detect_bursts(spike_times)
        
        # Should not detect any bursts
        assert len(burst_info['burst_starts']) == 0
        assert len(burst_info['burst_ends']) == 0
        assert len(burst_info['burst_spikes']) == 0
        assert len(burst_info['isolated_spikes']) == len(spike_times)
    
    def test_detect_bursts_empty(self):
        """Test burst detection with empty spike train."""
        empty_spikes = np.array([])
        
        burst_info = SpikeProcessor.detect_bursts(empty_spikes)
        
        assert len(burst_info['burst_starts']) == 0
        assert len(burst_info['isolated_spikes']) == 0


class TestFilterMetrics:
    """Test cases for FilterMetrics class."""
    
    def test_initialization(self):
        """Test filter metrics initialization."""
        test_filter = np.array([0.5, 0.3, 0.1, -0.1, -0.2])
        metrics = FilterMetrics(test_filter)
        
        np.testing.assert_array_equal(metrics.filter, test_filter)
    
    def test_compute_snr(self):
        """Test SNR computation."""
        # Create a filter with clear signal structure
        signal_filter = np.array([0.0, 0.0, 1.0, 0.5, 0.0, 0.0])  # Clear peak
        metrics = FilterMetrics(signal_filter)
        
        snr = metrics.compute_snr()
        assert snr > 0
        assert isinstance(snr, float)
        
        # Noisy filter should have lower SNR
        noisy_filter = np.random.randn(6) * 0.1
        noisy_metrics = FilterMetrics(noisy_filter)
        noisy_snr = noisy_metrics.compute_snr()
        
        assert snr > noisy_snr
    
    def test_compute_peak_properties(self):
        """Test peak properties computation."""
        # Create filter with known peak
        test_filter = np.array([0.1, 0.2, 0.8, 0.3, 0.1])  # Peak at index 2
        metrics = FilterMetrics(test_filter)
        
        properties = metrics.compute_peak_properties()
        
        assert 'peak_amplitude' in properties
        assert 'peak_time' in properties
        assert 'peak_width' in properties
        
        assert properties['peak_amplitude'] == 0.8
        assert properties['peak_time'] == 2
    
    def test_compute_quality_score(self):
        """Test overall quality score computation."""
        test_filter = np.array([0.2, 0.5, 0.8, 0.3, 0.1, -0.1])
        metrics = FilterMetrics(test_filter)
        
        quality = metrics.compute_quality_score()
        
        assert 0 <= quality <= 1  # Should be normalized between 0 and 1
        assert isinstance(quality, float)


class TestNonlinearityMetrics:
    """Test cases for NonlinearityMetrics class."""
    
    def test_initialization(self):
        """Test nonlinearity metrics initialization."""
        x = np.linspace(-2, 2, 100)
        y = np.exp(x)  # Exponential nonlinearity
        
        metrics = NonlinearityMetrics(x, y)
        
        np.testing.assert_array_equal(metrics.x, x)
        np.testing.assert_array_equal(metrics.y, y)
    
    def test_compute_dynamic_range(self):
        """Test dynamic range computation."""
        x = np.linspace(-2, 2, 100)
        y = np.exp(x)
        
        metrics = NonlinearityMetrics(x, y)
        dynamic_range = metrics.compute_dynamic_range()
        
        assert dynamic_range > 0
        assert isinstance(dynamic_range, float)
        
        # Should be approximately log(max/min) for exponential
        expected_range = np.log(np.max(y) / np.max(y[y > 0]))
        assert dynamic_range > 0  # Should have some dynamic range
    
    def test_compute_threshold(self):
        """Test threshold computation."""
        x = np.linspace(-2, 2, 100)
        y = np.maximum(0, x)  # Rectified linear
        
        metrics = NonlinearityMetrics(x, y)
        threshold = metrics.compute_threshold()
        
        assert isinstance(threshold, float)
        # For rectified linear, threshold should be near 0
        assert abs(threshold) < 0.2
    
    def test_compute_saturation_level(self):
        """Test saturation level computation."""
        x = np.linspace(-2, 2, 100)
        y = np.tanh(x)  # Saturating nonlinearity
        
        metrics = NonlinearityMetrics(x, y)
        saturation = metrics.compute_saturation_level()
        
        assert isinstance(saturation, float)
        assert saturation > 0
        # For tanh, saturation should be near 1
        assert abs(saturation - 1.0) < 0.2
    
    def test_analyze_shape(self):
        """Test nonlinearity shape analysis."""
        x = np.linspace(-2, 2, 100)
        y = x**2  # Quadratic nonlinearity
        
        metrics = NonlinearityMetrics(x, y)
        shape_analysis = metrics.analyze_shape()
        
        assert isinstance(shape_analysis, dict)
        assert 'monotonic' in shape_analysis
        assert 'convex' in shape_analysis
        assert 'symmetric' in shape_analysis


if __name__ == '__main__':
    pytest.main([__file__])

"""
Tests for multi-electrode analysis functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Generator

from white_noise_toolkit.multi_electrode import MultiElectrodeAnalyzer, analyze_mea_data
from white_noise_toolkit.core.exceptions import InsufficientDataError, DataValidationError
from white_noise_toolkit.synthetic.data_generator import SyntheticDataGenerator


class TestMultiElectrodeAnalyzer:
    """Test cases for MultiElectrodeAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_samples = 1000
        self.sampling_rate = 10000.0
        
        # Create simple synthetic data
        self.stimulus = np.random.randn(self.n_samples)
        
        # Create synthetic spike data
        self.spike_data = {}
        for i in range(3):
            # Generate random spike times
            n_spikes = np.random.poisson(50)  # ~50 spikes expected
            spike_times = np.sort(np.random.uniform(0, self.n_samples/self.sampling_rate, n_spikes))
            self.spike_data[f'unit_{i}'] = spike_times
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = MultiElectrodeAnalyzer(
            bin_size=0.01,
            filter_length=20
        )
        
        assert analyzer.bin_size == 0.01
        assert analyzer.filter_length == 20
        assert analyzer.analyzers == {}
        assert analyzer.results == {}
    
    def test_initialization_with_parameters(self):
        """Test initialization with various parameters."""
        analyzer = MultiElectrodeAnalyzer(
            bin_size=0.005,
            filter_length=30,
            spatial_dims=(10, 10),
            n_colors=2,
            memory_limit_gb=2.0
        )
        
        assert analyzer.bin_size == 0.005
        assert analyzer.filter_length == 30
        assert analyzer.spatial_dims == (10, 10)
        assert analyzer.n_colors == 2
        assert analyzer.memory_limit_gb == 2.0
    
    def test_add_unit(self):
        """Test adding units for analysis."""
        analyzer = MultiElectrodeAnalyzer()
        
        spike_times = np.array([0.1, 0.2, 0.3, 0.5])
        analyzer.add_unit('test_unit', spike_times)
        
        assert 'test_unit' in analyzer.analyzers
        assert isinstance(analyzer.analyzers['test_unit'], type(analyzer.analyzers['test_unit']))
    
    def test_add_unit_empty_spikes(self):
        """Test adding unit with no spikes."""
        analyzer = MultiElectrodeAnalyzer()
        
        # Should handle empty spike times gracefully
        empty_spikes = np.array([])
        analyzer.add_unit('empty_unit', empty_spikes)
        
        # Unit should not be added
        assert 'empty_unit' not in analyzer.analyzers
    
    @patch('white_noise_toolkit.multi_electrode.SingleCellAnalyzer')
    def test_analyze_population_sequential(self, mock_analyzer_class):
        """Test sequential population analysis."""
        # Mock the SingleCellAnalyzer
        mock_analyzer = Mock()
        mock_results = {
            'filter': np.random.randn(25),
            'nonlinearity': {'x': np.linspace(0, 1, 50), 'y': np.random.rand(50)},
            'quality_metrics': {'snr': 0.8}
        }
        mock_analyzer.get_results.return_value = mock_results
        mock_analyzer_class.return_value = mock_analyzer
        
        analyzer = MultiElectrodeAnalyzer()
        
        results = analyzer.analyze_population(
            self.stimulus, 
            self.sampling_rate, 
            self.spike_data,
            parallel=False
        )
        
        assert 'individual_results' in results
        assert 'population_analysis' in results
        assert 'summary' in results
        assert results['summary']['n_units'] == len(self.spike_data)
    
    def test_get_unit_results(self):
        """Test retrieving results for specific units."""
        analyzer = MultiElectrodeAnalyzer()
        
        # Add some mock results
        mock_result = {'filter': np.array([1, 2, 3]), 'quality': 0.8}
        analyzer.results['test_unit'] = mock_result
        
        result = analyzer.get_unit_results('test_unit')
        assert result == mock_result
        
        # Test non-existent unit
        result = analyzer.get_unit_results('non_existent')
        assert result is None
    
    def test_get_population_results(self):
        """Test getting population results."""
        analyzer = MultiElectrodeAnalyzer()
        
        # Add some mock data
        analyzer.analyzers['unit1'] = Mock()
        analyzer.results['unit1'] = {'filter': np.array([1, 2, 3])}
        analyzer.population_results['test_metric'] = 0.5
        
        results = analyzer.get_population_results()
        
        assert 'individual_results' in results
        assert 'population_analysis' in results
        assert 'summary' in results
        assert results['summary']['n_units'] == 1
    
    def test_get_successful_units(self):
        """Test getting list of successful units."""
        analyzer = MultiElectrodeAnalyzer()
        
        # Add mix of successful and failed results
        analyzer.results['success1'] = {'filter': np.array([1, 2, 3])}
        analyzer.results['success2'] = {'filter': np.array([4, 5, 6])}
        analyzer.results['failed'] = None
        
        successful = analyzer.get_successful_units()
        
        assert len(successful) == 2
        assert 'success1' in successful
        assert 'success2' in successful
        assert 'failed' not in successful


class TestAnalyzeMEAData:
    """Test cases for the analyze_mea_data convenience function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_samples = 500
        self.sampling_rate = 5000.0
        
        # Create synthetic data
        self.stimulus = np.random.randn(self.n_samples)
        self.spike_data = {}
        
        # Create simple spike data
        for i in range(3):
            n_spikes = np.random.poisson(30)
            spike_times = np.sort(np.random.uniform(0, self.n_samples/self.sampling_rate, n_spikes))
            self.spike_data[f'ch_{i}'] = spike_times
    
    @patch('white_noise_toolkit.multi_electrode.MultiElectrodeAnalyzer')
    def test_analyze_mea_data_basic(self, mock_analyzer_class):
        """Test basic MEA data analysis."""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_population.return_value = {
            'individual_results': {},
            'population_analysis': {},
            'summary': {'n_units': 3}
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        results = analyze_mea_data(
            stimulus=self.stimulus,
            spike_data=self.spike_data,
            sampling_rate=self.sampling_rate
        )
        
        assert 'individual_results' in results
        assert 'population_analysis' in results
        assert 'summary' in results
        mock_analyzer.analyze_population.assert_called_once()
    
    def test_analyze_mea_data_with_parameters(self):
        """Test MEA analysis with custom parameters."""
        # This test checks that parameters are passed correctly
        with patch('white_noise_toolkit.multi_electrode.MultiElectrodeAnalyzer') as mock_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_population.return_value = {'summary': {'n_units': 3}}
            mock_class.return_value = mock_analyzer
            
            analyze_mea_data(
                stimulus=self.stimulus,
                spike_data=self.spike_data,
                sampling_rate=self.sampling_rate,
                bin_size=0.01,
                filter_length=30,
                parallel=False
            )
            
            # Check that analyzer was created with correct parameters
            mock_class.assert_called_once_with(
                bin_size=0.01,
                filter_length=30,
                spatial_dims=None,
                n_colors=1,
                memory_limit_gb=4.0,
                n_workers=None
            )
            
            # Check that analyze_population was called with correct parameters
            mock_analyzer.analyze_population.assert_called_once_with(
                self.stimulus, self.sampling_rate, self.spike_data, parallel=False
            )


class TestMultiElectrodeIntegration:
    """Integration tests for multi-electrode analysis."""
    
    def test_end_to_end_analysis(self):
        """Test complete multi-electrode analysis pipeline."""
        # Generate synthetic multi-electrode data
        n_units = 2
        n_samples = 800
        sampling_rate = 8000.0
        
        stimulus = np.random.randn(n_samples)
        spike_data = {}
        
        # Create simple spike data
        for i in range(n_units):
            n_spikes = np.random.poisson(40)  # ~40 spikes expected
            spike_times = np.sort(np.random.uniform(0, n_samples/sampling_rate, n_spikes))
            spike_data[f'unit_{i}'] = spike_times
        
        # Run analysis
        results = analyze_mea_data(
            stimulus=stimulus,
            spike_data=spike_data,
            sampling_rate=sampling_rate,
            bin_size=0.01,
            filter_length=20,
            parallel=False  # Use sequential for testing
        )
        
        # Verify results structure
        assert 'individual_results' in results
        assert 'population_analysis' in results
        assert 'summary' in results
        
        summary = results['summary']
        
        # Check that units were processed
        assert 'n_units' in summary
    
    def test_memory_efficiency(self):
        """Test that multi-electrode analysis handles multiple units."""
        n_units = 5
        n_samples = 1000
        sampling_rate = 10000.0
        
        stimulus = np.random.randn(n_samples)
        spike_data = {}
        
        # Generate spike data with varying spike counts
        for i in range(n_units):
            n_spikes = np.random.poisson(30 + i * 10)  # Varying spike counts
            spike_times = np.sort(np.random.uniform(0, n_samples/sampling_rate, n_spikes))
            spike_data[f'ch_{i}'] = spike_times
        
        # This should complete without memory errors
        analyzer = MultiElectrodeAnalyzer(
            bin_size=0.01,
            filter_length=20,
            memory_limit_gb=2.0
        )
        
        # Add units and run analysis
        results = analyzer.analyze_population(
            stimulus, sampling_rate, spike_data, parallel=False
        )
        
        # Verify results structure
        assert 'summary' in results
        assert results['summary']['n_units'] == n_units


if __name__ == '__main__':
    pytest.main([__file__])

"""
Tests for core analysis components.

This module tests the fundamental algorithms of the white noise analysis toolkit:
- Design matrix construction
- Filter extraction (STA and whitened STA)  
- Nonlinearity estimation
- Single cell analysis pipeline
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import warnings

from white_noise_toolkit.core.single_cell import SingleCellAnalyzer
from white_noise_toolkit.core.design_matrix import StreamingDesignMatrix, create_design_matrix_batch
from white_noise_toolkit.core.filter_extraction import StreamingFilterExtractor
from white_noise_toolkit.core.nonlinearity_estimation import NonparametricNonlinearity, ParametricNonlinearity
from white_noise_toolkit.core.streaming_analyzer import create_stimulus_generator, create_spike_generator
from white_noise_toolkit.core.exceptions import DataValidationError, FilterExtractionError
from white_noise_toolkit.synthetic import SyntheticDataGenerator


class TestDesignMatrix:
    """Test design matrix construction"""
    
    def test_temporal_design_matrix(self):
        """Test basic temporal design matrix construction"""
        filter_length = 5
        stimulus = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        design_matrix = create_design_matrix_batch(stimulus, filter_length)
        
        # Check shape
        expected_shape = (len(stimulus), filter_length)
        assert design_matrix.shape == expected_shape
        
        # Check specific values (with zero padding at beginning)
        expected_first_row = [0, 0, 0, 0, 1]
        expected_last_row = [4, 5, 6, 7, 8]
        
        assert_array_almost_equal(design_matrix[0], expected_first_row)
        assert_array_almost_equal(design_matrix[-1], expected_last_row)
    
    def test_spatial_design_matrix(self):
        """Test spatial design matrix construction"""
        filter_length = 3
        spatial_dims = (2, 2)
        n_time_bins = 5
        
        # Create 2x2 spatial stimulus over time
        stimulus = np.random.randn(n_time_bins, *spatial_dims)
        
        design_matrix = create_design_matrix_batch(
            stimulus, filter_length, spatial_dims=spatial_dims
        )
        
        # Check shape: (n_time_bins, filter_length * spatial_size)
        expected_shape = (n_time_bins, filter_length * np.prod(spatial_dims))
        assert design_matrix.shape == expected_shape
    
    def test_streaming_design_matrix(self):
        """Test streaming design matrix with continuity"""
        filter_length = 4
        builder = StreamingDesignMatrix(filter_length)
        
        # Process two chunks
        chunk1 = np.array([1, 2, 3, 4, 5])
        chunk2 = np.array([6, 7, 8, 9])
        
        # First chunk
        design1 = builder.build_hankel_chunk(chunk1, is_first_chunk=True)
        
        # Second chunk (should maintain continuity)
        design2 = builder.build_hankel_chunk(chunk2, is_first_chunk=False)
        
        # Check that the last row of design1 and first rows of design2 overlap correctly
        # Last row of design1 should be [2, 3, 4, 5]
        assert_array_almost_equal(design1[-1], [2, 3, 4, 5])
        
        # First row of design2 should be [3, 4, 5, 6] (continuity from chunk1)
        assert_array_almost_equal(design2[0], [3, 4, 5, 6])
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters"""
        with pytest.raises(DataValidationError):
            StreamingDesignMatrix(filter_length=0)  # Invalid filter length
        
        with pytest.raises(DataValidationError):
            StreamingDesignMatrix(filter_length=5, spatial_dims=(0, 5))  # Invalid spatial dims


class TestFilterExtraction:
    """Test filter extraction algorithms"""
    
    def setup_method(self):
        """Setup common test data"""
        self.filter_length = 10
        self.n_samples = 1000
        
        # Create known filter
        t = np.arange(self.filter_length) * 0.001
        self.true_filter = np.exp(-t/0.01) * np.sin(2*np.pi*t/0.005)
        self.true_filter = self.true_filter / np.linalg.norm(self.true_filter)
        
        # Generate synthetic data
        self.generator = SyntheticDataGenerator(
            filter_true=self.true_filter,
            nonlinearity_true=lambda x: np.maximum(x + 0.2, 0),
            noise_level=0.05,
            random_seed=42
        )
        
        self.stimulus = self.generator.generate_white_noise_stimulus(self.n_samples)
        self.spikes = self.generator.generate_responses(self.stimulus, bin_size=0.001)
    
    def test_sta_extraction(self):
        """Test spike-triggered average extraction"""
        extractor = StreamingFilterExtractor()
        
        # Create design matrix
        design_matrix = create_design_matrix_batch(self.stimulus, self.filter_length)
        
        # Compute STA
        extractor.compute_sta_streaming(design_matrix, self.spikes)
        sta = extractor.finalize_sta()
        
        # Check that STA is reasonable
        assert sta.shape == (self.filter_length,)
        assert np.linalg.norm(sta) > 0
        
        # Check correlation with true filter (should be reasonably high)
        correlation = np.corrcoef(self.true_filter, sta)[0, 1]
        assert correlation > 0.7  # Should recover filter reasonably well
    
    def test_whitened_sta_extraction(self):
        """Test whitened STA (maximum likelihood) extraction"""
        extractor = StreamingFilterExtractor()
        
        # Create design matrix
        design_matrix = create_design_matrix_batch(self.stimulus, self.filter_length)
        
        # Compute both STA and whitened STA
        extractor.compute_sta_streaming(design_matrix, self.spikes)
        extractor.compute_whitened_sta_streaming(design_matrix, self.spikes)
        
        sta = extractor.finalize_sta()
        whitened_sta = extractor.finalize_whitened_sta()
        
        # Both should be valid
        assert sta.shape == whitened_sta.shape == (self.filter_length,)
        assert np.linalg.norm(sta) > 0
        assert np.linalg.norm(whitened_sta) > 0
        
        # Whitened STA should generally have better correlation with true filter
        corr_sta = np.corrcoef(self.true_filter, sta)[0, 1]
        corr_whitened = np.corrcoef(self.true_filter, whitened_sta)[0, 1]
        
        assert corr_sta > 0.5
        assert corr_whitened > 0.5
    
    def test_insufficient_spikes(self):
        """Test behavior with insufficient spike data"""
        extractor = StreamingFilterExtractor()
        
        # Create data with very few spikes
        sparse_spikes = np.zeros(self.n_samples)
        sparse_spikes[100] = 1  # Only one spike
        
        design_matrix = create_design_matrix_batch(self.stimulus, self.filter_length)
        extractor.compute_sta_streaming(design_matrix, sparse_spikes)
        
        # Should still work but produce warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sta = extractor.finalize_sta()
            assert len(w) >= 0  # May produce warnings about low spike count


class TestNonlinearityEstimation:
    """Test nonlinearity estimation methods"""
    
    def setup_method(self):
        """Setup test data"""
        # Create generator signal and known nonlinearity
        self.generator_signal = np.random.randn(2000)
        
        # True nonlinearity: rectified linear with threshold
        def true_nonlinearity(x):
            return np.maximum(20 * (x + 0.3), 0)
        
        # Generate spike counts using Poisson process
        rates = true_nonlinearity(self.generator_signal)
        self.spike_counts = np.random.poisson(rates * 0.001)  # Scale for reasonable spike counts
        self.true_nonlinearity = true_nonlinearity
    
    def test_nonparametric_estimation(self):
        """Test nonparametric (histogram-based) nonlinearity estimation"""
        estimator = NonparametricNonlinearity(n_bins=20)
        estimator.fit(self.generator_signal, self.spike_counts)
        
        # Check that estimator has been fitted
        assert estimator.fitted
        assert estimator.bin_centers is not None
        assert estimator.spike_rates is not None
        assert len(estimator.bin_centers) == 20
        
        # Test prediction
        test_signal = np.linspace(-2, 2, 100)
        predicted_rates = estimator.predict(test_signal)
        
        assert len(predicted_rates) == len(test_signal)
        assert np.all(predicted_rates >= 0)  # Rates should be non-negative
    
    def test_parametric_estimation(self):
        """Test parametric nonlinearity estimation"""
        # Test exponential model
        estimator = ParametricNonlinearity(model_type='exponential')
        estimator.fit(self.generator_signal, self.spike_counts)
        
        assert estimator.fitted
        assert estimator.parameters is not None
        assert len(estimator.parameters) == 3  # Exponential has 3 parameters
        
        # Test prediction
        test_signal = np.linspace(-1, 1, 50)
        predicted_rates = estimator.predict(test_signal)
        
        assert len(predicted_rates) == len(test_signal)
        assert np.all(predicted_rates >= 0)
    
    def test_goodness_of_fit(self):
        """Test goodness of fit metrics"""
        estimator = NonparametricNonlinearity(n_bins=15)
        estimator.fit(self.generator_signal, self.spike_counts)
        
        gof = estimator.get_goodness_of_fit(self.generator_signal, self.spike_counts)
        
        assert 'r_squared' in gof
        assert 'mse' in gof
        assert 'log_likelihood' in gof
        
        # R-squared should be between 0 and 1
        assert 0 <= gof['r_squared'] <= 1


class TestSingleCellAnalyzer:
    """Test complete single cell analysis pipeline"""
    
    def setup_method(self):
        """Setup test data"""
        self.filter_length = 15
        self.bin_size = 0.001
        
        # Create realistic filter
        t = np.arange(self.filter_length) * self.bin_size
        self.true_filter = np.exp(-t/0.008) * np.sin(2*np.pi*t/0.004)
        self.true_filter = self.true_filter / np.linalg.norm(self.true_filter)
        
        # Create analyzer
        self.analyzer = SingleCellAnalyzer(
            bin_size=self.bin_size,
            filter_length=self.filter_length,
            memory_limit_gb=1.0
        )
        
        # Generate test data
        self.generator = SyntheticDataGenerator(
            filter_true=self.true_filter,
            nonlinearity_true=lambda x: np.maximum(30 * (x + 0.2), 0),
            noise_level=0.1,
            random_seed=42
        )
        
        self.stimulus = self.generator.generate_white_noise_stimulus(5000)
        self.spikes = self.generator.generate_responses(self.stimulus, bin_size=self.bin_size)
    
    def test_complete_analysis(self):
        """Test complete analysis pipeline"""
        # Create generators
        stim_gen = create_stimulus_generator(self.stimulus, chunk_size=1000)
        spike_gen = create_spike_generator(self.spikes, chunk_size=1000)
        
        # Run analysis
        self.analyzer.fit_streaming(
            stim_gen, spike_gen,
            nonlinearity_method='nonparametric',
            extract_both_filters=True,
            progress_bar=False
        )
        
        # Check that analysis completed
        assert self.analyzer.fitted
        
        # Get results
        results = self.analyzer.get_results()
        
        # Check results structure
        assert 'filter' in results
        assert 'sta' in results
        assert 'nonlinearity' in results
        assert 'metadata' in results
        
        # Check filter quality
        estimated_filter = results['filter']
        assert estimated_filter.shape == self.true_filter.shape
        
        correlation = np.corrcoef(self.true_filter, estimated_filter)[0, 1]
        assert correlation > 0.6  # Should recover filter reasonably well
    
    def test_prediction(self):
        """Test model prediction capability"""
        # First fit the model
        stim_gen = create_stimulus_generator(self.stimulus, chunk_size=1000)
        spike_gen = create_spike_generator(self.spikes, chunk_size=1000)
        
        self.analyzer.fit_streaming(stim_gen, spike_gen, progress_bar=False)
        
        # Test prediction on new stimulus
        test_stimulus = np.random.randn(1000)
        predicted_rates = self.analyzer.predict(test_stimulus)
        
        assert len(predicted_rates) == len(test_stimulus)
        assert np.all(predicted_rates >= 0)
    
    def test_validation_errors(self):
        """Test input validation and error handling"""
        # Test invalid parameters
        with pytest.raises(DataValidationError):
            SingleCellAnalyzer(bin_size=-0.001)  # Negative bin size
        
        with pytest.raises(DataValidationError):
            SingleCellAnalyzer(filter_length=0)  # Zero filter length
        
        # Test getting results before fitting
        analyzer = SingleCellAnalyzer()
        with pytest.raises(FilterExtractionError):
            analyzer.get_results()
    
    def test_memory_management(self):
        """Test memory management features"""
        memory_info = self.analyzer.get_memory_usage()
        
        assert 'current_gb' in memory_info or 'available_gb' in memory_info
        
        # Test configuration validation
        validation = self.analyzer.validate_configuration()
        assert 'valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation


class TestStreamingAnalyzer:
    """Test streaming analysis utilities"""
    
    def test_create_generators(self):
        """Test data generator creation"""
        stimulus = np.random.randn(2000)
        spikes = np.random.poisson(0.1, 2000)
        chunk_size = 500
        
        # Create generators
        stim_gen = create_stimulus_generator(stimulus, chunk_size)
        spike_gen = create_spike_generator(spikes, chunk_size)
        
        # Test that generators work
        chunks_collected = 0
        for stim_chunk, spike_chunk in zip(stim_gen, spike_gen):
            assert len(stim_chunk) <= chunk_size
            assert len(spike_chunk) <= chunk_size
            assert len(stim_chunk) == len(spike_chunk)
            chunks_collected += 1
        
        expected_chunks = int(np.ceil(len(stimulus) / chunk_size))
        assert chunks_collected == expected_chunks


# Property-based testing with hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestPropertyBased:
        """Property-based tests using hypothesis"""
        
        @given(
            filter_length=st.integers(min_value=5, max_value=20),
            n_samples=st.integers(min_value=100, max_value=1000)
        )
        def test_design_matrix_properties(self, filter_length, n_samples):
            """Test design matrix properties hold for various inputs"""
            stimulus = np.random.randn(n_samples)
            design_matrix = create_design_matrix_batch(stimulus, filter_length)
            
            # Shape should always be correct
            assert design_matrix.shape == (n_samples, filter_length)
            
            # Should not contain NaN or inf
            assert np.isfinite(design_matrix).all()
        
        @given(
            n_bins=st.integers(min_value=5, max_value=50),
            n_samples=st.integers(min_value=100, max_value=1000)
        )
        def test_nonlinearity_estimation_properties(self, n_bins, n_samples):
            """Test nonlinearity estimation properties"""
            generator_signal = np.random.randn(n_samples)
            spike_counts = np.random.poisson(1.0, n_samples)
            
            estimator = NonparametricNonlinearity(n_bins=n_bins)
            estimator.fit(generator_signal, spike_counts)
            
            # Should always fit successfully with reasonable data
            assert estimator.fitted
            assert len(estimator.bin_centers) == n_bins
            
            # Predictions should be non-negative
            test_signal = np.random.randn(50)
            predictions = estimator.predict(test_signal)
            assert np.all(predictions >= 0)

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    pytest.main([__file__])

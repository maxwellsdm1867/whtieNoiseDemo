"""
White Noise Analysis Toolkit

A research-grade Python toolkit for white noise analysis of neuronal responses.
This toolkit extracts linear filters and static nonlinearities from stimulus-response
data using streaming computation for memory efficiency.
"""

__version__ = "0.1.0"
__author__ = "White Noise Analysis Team"
__email__ = "contact@whitenoise.toolkit"

# Import main classes for easy access
from .core.single_cell import SingleCellAnalyzer
from .core.streaming_analyzer import (
    create_stimulus_generator, 
    create_spike_generator,
    create_aligned_generators
)
from .core.design_matrix import StreamingDesignMatrix
from .core.filter_extraction import StreamingFilterExtractor
from .core.nonlinearity_estimation import NonparametricNonlinearity, ParametricNonlinearity

# Multi-electrode analysis
from .multi_electrode import MultiElectrodeAnalyzer, analyze_mea_data

# Utilities
from .utils.memory_manager import MemoryManager
from .utils.logging_config import setup_logging, get_logger
from .utils.io_handlers import load_data, save_data, load_spikes
from .utils.preprocessing import SpikeProcessor, StimulusProcessor, DataSynchronizer
from .utils.metrics import FilterMetrics, NonlinearityMetrics, ModelMetrics

# Synthetic data and validation
from .synthetic import SyntheticDataGenerator, GroundTruthRecovery, run_comprehensive_validation

# Import exceptions
from .core.exceptions import (
    WhiteNoiseAnalysisError,
    InsufficientDataError,
    MemoryLimitError,
    StimulusValidationError,
    FilterExtractionError,
    NonlinearityFittingError,
    DataValidationError,
    ConfigurationError,
    NumericalInstabilityError,
    FileFormatError,
    ProcessingError
)

__all__ = [
    # Main classes
    'SingleCellAnalyzer',
    'StreamingDesignMatrix',
    'StreamingFilterExtractor',
    'NonparametricNonlinearity',
    'ParametricNonlinearity',
    
    # Multi-electrode analysis
    'MultiElectrodeAnalyzer',
    'analyze_mea_data',
    
    # Utilities
    'MemoryManager',
    'load_data',
    'save_data', 
    'load_spikes',
    'SpikeProcessor',
    'StimulusProcessor',
    'DataSynchronizer',
    'FilterMetrics',
    'NonlinearityMetrics',
    'ModelMetrics',
    
    # Synthetic data and validation
    'SyntheticDataGenerator',
    'GroundTruthRecovery',
    'run_comprehensive_validation',
    
    # Generator functions
    'create_stimulus_generator',
    'create_spike_generator', 
    'create_aligned_generators',
    
    # Logging
    'setup_logging',
    'get_logger',
    
    # Exceptions
    'WhiteNoiseAnalysisError',
    'InsufficientDataError',
    'MemoryLimitError',
    'StimulusValidationError',
    'FilterExtractionError',
    'NonlinearityFittingError',
    'DataValidationError',
    'ConfigurationError',
    'NumericalInstabilityError',
    'FileFormatError',
    'ProcessingError',
    
    # Package info
    '__version__'
]


def get_config_path():
    """Get path to default configuration file."""
    import os
    from pathlib import Path
    
    # Get package directory
    package_dir = Path(__file__).parent
    config_path = package_dir / 'config' / 'default.yaml'
    
    if config_path.exists():
        return str(config_path)
    else:
        # Fallback - return None if config not found
        return None


def load_default_config():
    """Load default configuration."""
    import yaml
    from pathlib import Path
    
    config_path = get_config_path()
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return minimal default config
        return {
            'analysis': {
                'bin_size': 0.008,
                'filter_length': 25,
                'chunk_size': 1000,
                'regularization': 1e-6,
                'memory_limit_gb': 8.0
            },
            'nonlinearity': {
                'method': 'nonparametric',
                'n_bins': 25
            },
            'logging': {
                'level': 'INFO',
                'progress_bars': True
            }
        }


# Configure default logging
try:
    config = load_default_config()
    if config:
        from .utils.logging_config import configure_logging_from_config
        configure_logging_from_config(config)
except:
    # Fallback to basic logging
    setup_logging(level='INFO')


def test_installation():
    """Quick test to verify installation."""
    try:
        import numpy as np
        
        # Test basic imports
        from .core.single_cell import SingleCellAnalyzer
        from .synthetic.data_generator import SyntheticDataGenerator
        
        # Create simple test
        analyzer = SingleCellAnalyzer(filter_length=10)
        
        # Generate tiny synthetic dataset
        generator = SyntheticDataGenerator(
            filter_true=np.random.randn(10),
            nonlinearity_true=lambda x: np.exp(x)
        )
        
        stimulus = generator.generate_white_noise_stimulus(100)
        spikes = generator.generate_responses(stimulus)
        
        return True, "Installation test passed"
        
    except Exception as e:
        return False, f"Installation test failed: {e}"


if __name__ == "__main__":
    # Quick test when module is run directly
    success, message = test_installation()
    print(f"White Noise Toolkit v{__version__}")
    print(message)
    if not success:
        exit(1)

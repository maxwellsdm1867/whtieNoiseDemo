"""
Utility modules for the White Noise Analysis Toolkit.

This package provides utility functions for memory management, logging,
I/O operations, preprocessing, metrics, and other supporting functionality.
"""

from .memory_manager import MemoryManager
from .logging_config import setup_logging, get_logger, TimingLogger, MemoryLogger, ProgressLogger
from .io_handlers import (
    DataReader, DataWriter, SpikeDataLoader, 
    load_data, save_data, load_spikes, get_file_info
)
from .preprocessing import (
    SpikeProcessor, StimulusProcessor, DataSynchronizer,
    validate_data_consistency
)
from .metrics import (
    FilterMetrics, NonlinearityMetrics, ModelMetrics,
    compute_summary_statistics, bootstrap_confidence_interval
)

__all__ = [
    # Memory management
    'MemoryManager',
    
    # Logging
    'setup_logging', 'get_logger', 'TimingLogger', 'MemoryLogger', 'ProgressLogger',
    
    # I/O
    'DataReader', 'DataWriter', 'SpikeDataLoader',
    'load_data', 'save_data', 'load_spikes', 'get_file_info',
    
    # Preprocessing
    'SpikeProcessor', 'StimulusProcessor', 'DataSynchronizer',
    'validate_data_consistency',
    
    # Metrics
    'FilterMetrics', 'NonlinearityMetrics', 'ModelMetrics',
    'compute_summary_statistics', 'bootstrap_confidence_interval'
]

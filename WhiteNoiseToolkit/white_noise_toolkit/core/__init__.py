"""Core module for white noise analysis."""

from .single_cell import SingleCellAnalyzer
from .streaming_analyzer import (
    create_stimulus_generator,
    create_spike_generator, 
    create_aligned_generators,
    validate_generators
)
from .design_matrix import StreamingDesignMatrix, create_design_matrix_batch
from .filter_extraction import StreamingFilterExtractor, compare_filter_methods, validate_filter_quality
from .nonlinearity_estimation import NonparametricNonlinearity, ParametricNonlinearity, compare_nonlinearity_models
from .exceptions import *

__all__ = [
    'SingleCellAnalyzer',
    'StreamingDesignMatrix',
    'StreamingFilterExtractor', 
    'NonparametricNonlinearity',
    'ParametricNonlinearity',
    'create_stimulus_generator',
    'create_spike_generator',
    'create_aligned_generators',
    'validate_generators',
    'create_design_matrix_batch',
    'compare_filter_methods',
    'validate_filter_quality',
    'compare_nonlinearity_models'
]

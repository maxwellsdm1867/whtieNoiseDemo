"""
Synthetic data generation and validation for the White Noise Analysis Toolkit.

This package provides tools for generating synthetic stimulus-response data
with known ground truth and validating analysis results.
"""

from .data_generator import SyntheticDataGenerator
from .validation import (
    GroundTruthRecovery, ParameterSweep,
    create_validation_report, run_comprehensive_validation
)

__all__ = [
    'SyntheticDataGenerator',
    'GroundTruthRecovery',
    'ParameterSweep',
    'create_validation_report',
    'run_comprehensive_validation'
]

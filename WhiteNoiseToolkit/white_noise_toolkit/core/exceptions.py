"""
Exception classes for the White Noise Analysis Toolkit.

This module defines custom exceptions used throughout the toolkit to provide
specific error handling for different failure modes in the analysis pipeline.
"""

from typing import Optional


class WhiteNoiseAnalysisError(Exception):
    """Base exception for toolkit."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        """
        Initialize base exception.
        
        Parameters
        ----------
        message : str
            Main error message
        details : str, optional
            Additional details about the error
        """
        super().__init__(message)
        self.message = message
        self.details = details
        
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class InsufficientDataError(WhiteNoiseAnalysisError):
    """Raised when not enough spikes/data for reliable estimation."""
    
    def __init__(self, message: str, spike_count: Optional[int] = None, 
                 min_required: Optional[int] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        spike_count : int, optional
            Actual number of spikes
        min_required : int, optional
            Minimum required spikes
        """
        if spike_count is not None and min_required is not None:
            details = f"Found {spike_count} spikes, need at least {min_required}"
        else:
            details = None
        super().__init__(message, details)
        self.spike_count = spike_count
        self.min_required = min_required


class MemoryLimitError(WhiteNoiseAnalysisError):
    """Raised when memory requirements exceed limits."""
    
    def __init__(self, message: str, required_gb: Optional[float] = None,
                 available_gb: Optional[float] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        required_gb : float, optional
            Memory required in GB
        available_gb : float, optional
            Available memory in GB
        """
        if required_gb is not None and available_gb is not None:
            details = f"Requires {required_gb:.2f} GB, but only {available_gb:.2f} GB available"
        else:
            details = None
        super().__init__(message, details)
        self.required_gb = required_gb
        self.available_gb = available_gb


class StimulusValidationError(WhiteNoiseAnalysisError):
    """Raised when stimulus doesn't meet white noise assumptions."""
    
    def __init__(self, message: str, validation_results: Optional[dict] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        validation_results : dict, optional
            Dictionary with validation statistics
        """
        if validation_results:
            details = f"Validation results: {validation_results}"
        else:
            details = None
        super().__init__(message, details)
        self.validation_results = validation_results


class FilterExtractionError(WhiteNoiseAnalysisError):
    """Raised when filter extraction fails."""
    
    def __init__(self, message: str, condition_number: Optional[float] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        condition_number : float, optional
            Condition number of design matrix
        """
        if condition_number is not None:
            details = f"Design matrix condition number: {condition_number:.2e}"
        else:
            details = None
        super().__init__(message, details)
        self.condition_number = condition_number


class NonlinearityFittingError(WhiteNoiseAnalysisError):
    """Raised when nonlinearity fitting fails."""
    
    def __init__(self, message: str, fitting_method: Optional[str] = None,
                 convergence_info: Optional[dict] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        fitting_method : str, optional
            Method used for fitting ('parametric' or 'nonparametric')
        convergence_info : dict, optional
            Information about convergence failure
        """
        details_parts = []
        if fitting_method:
            details_parts.append(f"Method: {fitting_method}")
        if convergence_info:
            details_parts.append(f"Convergence info: {convergence_info}")
        
        details = "; ".join(details_parts) if details_parts else None
        super().__init__(message, details)
        self.fitting_method = fitting_method
        self.convergence_info = convergence_info


class DataValidationError(WhiteNoiseAnalysisError):
    """Raised when input data validation fails."""
    
    def __init__(self, message: str, expected_shape: Optional[tuple] = None,
                 actual_shape: Optional[tuple] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        expected_shape : tuple, optional
            Expected data shape
        actual_shape : tuple, optional
            Actual data shape
        """
        if expected_shape is not None and actual_shape is not None:
            details = f"Expected shape {expected_shape}, got {actual_shape}"
        else:
            details = None
        super().__init__(message, details)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class ConfigurationError(WhiteNoiseAnalysisError):
    """Raised when configuration parameters are invalid."""
    
    def __init__(self, message: str, parameter: Optional[str] = None,
                 value: Optional[any] = None, valid_range: Optional[str] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        parameter : str, optional
            Name of invalid parameter
        value : any, optional
            Invalid value
        valid_range : str, optional
            Description of valid range
        """
        details_parts = []
        if parameter:
            details_parts.append(f"Parameter: {parameter}")
        if value is not None:
            details_parts.append(f"Value: {value}")
        if valid_range:
            details_parts.append(f"Valid range: {valid_range}")
            
        details = "; ".join(details_parts) if details_parts else None
        super().__init__(message, details)
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range


class NumericalInstabilityError(WhiteNoiseAnalysisError):
    """Raised when numerical computations become unstable."""
    
    def __init__(self, message: str, computation: Optional[str] = None,
                 suggestion: Optional[str] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        computation : str, optional
            Name of computation that failed
        suggestion : str, optional
            Suggested fix
        """
        details_parts = []
        if computation:
            details_parts.append(f"Computation: {computation}")
        if suggestion:
            details_parts.append(f"Suggestion: {suggestion}")
            
        details = "; ".join(details_parts) if details_parts else None
        super().__init__(message, details)
        self.computation = computation
        self.suggestion = suggestion


class FileFormatError(WhiteNoiseAnalysisError):
    """Raised when file format is unsupported or corrupted."""
    
    def __init__(self, message: str, filepath: Optional[str] = None, 
                 format: Optional[str] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        filepath : str, optional
            Path to the problematic file
        format : str, optional
            Expected or detected format
        """
        details = []
        if filepath:
            details.append(f"File: {filepath}")
        if format:
            details.append(f"Format: {format}")
        
        super().__init__(message, "; ".join(details) if details else None)
        self.filepath = filepath
        self.format = format


class ProcessingError(WhiteNoiseAnalysisError):
    """Raised when data processing fails."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 data_info: Optional[str] = None):
        """
        Parameters
        ----------
        message : str
            Error message
        operation : str, optional
            The processing operation that failed
        data_info : str, optional
            Information about the data being processed
        """
        details = []
        if operation:
            details.append(f"Operation: {operation}")
        if data_info:
            details.append(f"Data: {data_info}")
        
        super().__init__(message, "; ".join(details) if details else None)
        self.operation = operation
        self.data_info = data_info

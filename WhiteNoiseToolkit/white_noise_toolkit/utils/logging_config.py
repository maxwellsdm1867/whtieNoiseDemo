"""
Logging configuration for the White Noise Analysis Toolkit.

This module provides centralized logging configuration with support for
different log levels, file output, and performance monitoring.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union
import yaml
from ..core.exceptions import ConfigurationError


class TimingLogger:
    """Context manager for timing operations with logging."""

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        """
        Initialize timing logger.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        operation : str
            Description of operation being timed
        level : int
            Logging level
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.operation}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            if duration < 1:
                time_str = f"{duration*1000:.1f} ms"
            elif duration < 60:
                time_str = f"{duration:.1f} s"
            else:
                time_str = f"{duration/60:.1f} min"

            if exc_type is None:
                self.logger.log(self.level, f"Completed {self.operation} in {time_str}")
            else:
                self.logger.error(f"Failed {self.operation} after {time_str}: {exc_val}")


class MemoryLogger:
    """Logger for memory usage monitoring."""

    def __init__(self, logger: logging.Logger, memory_manager=None):
        """
        Initialize memory logger.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        memory_manager : MemoryManager, optional
            Memory manager instance
        """
        self.logger = logger
        self.memory_manager = memory_manager

    def log_memory_usage(self, context: str = "", level: int = logging.DEBUG):
        """
        Log current memory usage.

        Parameters
        ----------
        context : str
            Context description
        level : int
            Logging level
        """
        if self.memory_manager is not None:
            memory_info = self.memory_manager.get_memory_info()
            usage_gb = memory_info['current_usage_gb']
            usage_pct = memory_info['usage_fraction'] * 100

            context_str = f" ({context})" if context else ""
            self.logger.log(
                level,
                f"Memory usage{context_str}: {usage_gb:.1f} GB ({usage_pct:.1f}%)"
            )


class ProgressLogger:
    """Logger for progress tracking with optional memory monitoring."""

    def __init__(self, logger: logging.Logger, total_items: int,
                 operation: str = "Processing", log_interval: int = 10,
                 memory_manager=None):
        """
        Initialize progress logger.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        total_items : int
            Total number of items to process
        operation : str
            Description of operation
        log_interval : int
            Log progress every N items
        memory_manager : MemoryManager, optional
            Memory manager for usage monitoring
        """
        self.logger = logger
        self.total_items = total_items
        self.operation = operation
        self.log_interval = log_interval
        self.memory_manager = memory_manager
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.processed_items = 0

    def update(self, n_items: int = 1):
        """
        Update progress.

        Parameters
        ----------
        n_items : int
            Number of items processed
        """
        self.processed_items += n_items

        # Check if we should log progress
        if (self.processed_items % self.log_interval == 0 or
            self.processed_items >= self.total_items):

            current_time = time.time()
            elapsed = current_time - self.start_time
            progress_pct = (self.processed_items / self.total_items) * 100

            # Estimate remaining time
            if self.processed_items > 0:
                rate = self.processed_items / elapsed
                remaining_items = self.total_items - self.processed_items
                eta_seconds = remaining_items / rate if rate > 0 else 0

                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = f"{eta_seconds/60:.1f}min"
            else:
                eta_str = "unknown"

            # Log progress
            message = (f"{self.operation}: {self.processed_items}/{self.total_items} "
                      f"({progress_pct:.1f}%) - ETA: {eta_str}")

            # Add memory info if available
            if self.memory_manager is not None:
                memory_info = self.memory_manager.get_memory_info()
                memory_gb = memory_info['current_usage_gb']
                message += f" - Memory: {memory_gb:.1f}GB"

            self.logger.info(message)


def setup_logging(level: Union[str, int] = logging.INFO,
                 log_file: Optional[Union[str, Path]] = None,
                 format_string: Optional[str] = None,
                 include_memory: bool = True,
                 include_timing: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the toolkit.

    Parameters
    ----------
    level : str or int
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    log_file : str or Path, optional
        Path to log file. If None, logs only to console.
    format_string : str, optional
        Custom format string for log messages
    include_memory : bool
        Whether to include memory monitoring in logs
    include_timing : bool
        Whether to include timing information in logs

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Convert string level to integer
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create logger
    logger = logging.getLogger('white_noise_toolkit')
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Set default format
    if format_string is None:
        if include_timing:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent duplicate logs
    logger.propagate = False

    return logger


def load_logging_config_from_yaml(config_path: Union[str, Path]) -> dict:
    """
    Load logging configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Logging configuration

    Raises
    ------
    ConfigurationError
        If configuration file is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract logging configuration
        logging_config = config.get('logging', {})

        # Validate required fields
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level = logging_config.get('level', 'INFO')

        if level not in valid_levels:
            raise ConfigurationError(
                f"Invalid logging level: {level}",
                parameter='logging.level',
                value=level,
                valid_range=f"One of {valid_levels}"
            )

        return logging_config

    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML configuration: {e}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the toolkit.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses the calling module name.

    Returns
    -------
    logging.Logger
        Logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'white_noise_toolkit')
        else:
            name = 'white_noise_toolkit'

    # Ensure it's under the toolkit namespace
    if name and not name.startswith('white_noise_toolkit'):
        name = f'white_noise_toolkit.{name}'

    return logging.getLogger(name or 'white_noise_toolkit')


def configure_logging_from_config(config: dict) -> logging.Logger:
    """
    Configure logging from configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary with logging settings

    Returns
    -------
    logging.Logger
        Configured logger
    """
    logging_config = config.get('logging', {})

    return setup_logging(
        level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('log_file'),
        include_memory=logging_config.get('log_memory_usage', True),
        include_timing=logging_config.get('log_timing', True)
    )

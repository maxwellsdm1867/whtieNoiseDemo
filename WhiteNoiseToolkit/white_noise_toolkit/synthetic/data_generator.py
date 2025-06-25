"""
Synthetic data generation for the White Noise Analysis Toolkit.

This module provides tools for generating synthetic stimulus-response data
with known ground truth for testing and validation.
"""

from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import warnings
from ..core.exceptions import DataValidationError


class SyntheticDataGenerator:
    """
    Generate synthetic white noise stimulus and neuronal responses.

    This class generates synthetic data with known ground truth linear filters
    and nonlinearities for testing and validation of the analysis toolkit.
    """

    def __init__(self, filter_true: np.ndarray, nonlinearity_true: Callable[[np.ndarray], np.ndarray],
                 noise_level: float = 1.0, random_seed: Optional[int] = None):
        """
        Initialize synthetic data generator.

        Parameters
        ----------
        filter_true : np.ndarray
            True linear filter
        nonlinearity_true : Callable
            True nonlinearity function
        noise_level : float, default=1.0
            Level of multiplicative noise added to the firing rate.
            0.0 = no noise, 1.0 = standard noise level
        random_seed : int, optional
            Random seed for reproducibility

        Raises
        ------
        DataValidationError
            If parameters are invalid
        """
        if not callable(nonlinearity_true):
            raise DataValidationError("nonlinearity_true must be callable")

        if noise_level <= 0:
            raise DataValidationError(f"noise_level must be positive, got {noise_level}")

        self.filter_true = np.array(filter_true)
        self.nonlinearity_true = nonlinearity_true
        self.noise_level = noise_level
        self.random_seed = random_seed

        # Create a random number generator instance for reproducibility
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()

        # Validate filter
        if self.filter_true.size == 0:
            raise DataValidationError("filter_true cannot be empty")

        if not np.isfinite(self.filter_true).all():
            raise DataValidationError("filter_true contains non-finite values")

    def generate_white_noise_stimulus(self, n_time_bins: int,
                                    spatial_dims: Optional[Tuple[int, ...]] = None,
                                    n_colors: int = 1,
                                    contrast_std: float = 1.0) -> np.ndarray:
        """
        Generate Gaussian white noise stimulus.

        Parameters
        ----------
        n_time_bins : int
            Number of time bins
        spatial_dims : tuple, optional
            (height, width) for spatial stimuli
        n_colors : int, default=1
            Number of color channels
        contrast_std : float, default=1.0
            Standard deviation of stimulus contrast

        Returns
        -------
        np.ndarray
            White noise stimulus

        Raises
        ------
        DataValidationError
            If parameters are invalid
        """
        if n_time_bins <= 0:
            raise DataValidationError(f"n_time_bins must be positive, got {n_time_bins}")

        if contrast_std <= 0:
            raise DataValidationError(f"contrast_std must be positive, got {contrast_std}")

        if spatial_dims is None:
            # 1D temporal stimulus
            stimulus_shape = (n_time_bins,)
        else:
            # Spatial-temporal stimulus
            if len(spatial_dims) != 2:
                raise DataValidationError(f"spatial_dims must be 2-tuple, got {spatial_dims}")

            if n_colors == 1:
                stimulus_shape = (n_time_bins,) + spatial_dims
            else:
                stimulus_shape = (n_time_bins,) + spatial_dims + (n_colors,)

        # Generate white noise
        stimulus = self.rng.normal(0, contrast_std, stimulus_shape)

        return stimulus

    def generate_responses(self, stimulus: np.ndarray, bin_size: float = 0.008) -> np.ndarray:
        """
        Apply true filter and nonlinearity, add Poisson spiking noise.

        Parameters
        ----------
        stimulus : np.ndarray
            Stimulus array
        bin_size : float, default=0.008
            Temporal bin size in seconds

        Returns
        -------
        np.ndarray
            Spike counts per time bin

        Raises
        ------
        DataValidationError
            If stimulus is invalid
        """
        if stimulus.size == 0:
            raise DataValidationError("stimulus is empty")

        if bin_size <= 0:
            raise DataValidationError(f"bin_size must be positive, got {bin_size}")

        # Create design matrix
        from ..core.design_matrix import create_design_matrix_batch

        # For compatibility with 1D filters, vectorize spatial stimuli
        if stimulus.ndim > 1 and len(self.filter_true.shape) == 1:
            # We have a spatial stimulus but a 1D temporal filter
            # Vectorize the stimulus to 1D (flatten spatial dimensions)
            n_time_bins = stimulus.shape[0]
            spatial_size = np.prod(stimulus.shape[1:])
            
            # Reshape to (n_time_bins, spatial_size) then take mean across spatial
            stimulus_vectorized = stimulus.reshape(n_time_bins, -1).mean(axis=1)
            
            design_matrix = create_design_matrix_batch(
                stimulus_vectorized,
                filter_length=len(self.filter_true),
                spatial_dims=None,
                n_colors=1
            )
        else:
            # Standard case: use stimulus as-is
            if stimulus.ndim == 1:
                # 1D temporal stimulus
                spatial_dims = None
                n_colors = 1
            elif stimulus.ndim == 2:
                # 2D spatial stimulus (assuming monochrome)
                spatial_dims = stimulus.shape[1:]
                n_colors = 1
            elif stimulus.ndim == 3:
                # Could be 2D spatial or 2D spatial with colors
                if stimulus.shape[-1] <= 4:  # Assume last dim is colors
                    spatial_dims = stimulus.shape[1:-1]
                    n_colors = stimulus.shape[-1]
                else:  # Assume no color dimension
                    spatial_dims = stimulus.shape[1:]
                    n_colors = 1
            else:
                # 3D spatial with colors
                spatial_dims = stimulus.shape[1:-1]
                n_colors = stimulus.shape[-1]

            try:
                design_matrix = create_design_matrix_batch(
                    stimulus,
                    filter_length=len(self.filter_true),
                    spatial_dims=spatial_dims,
                    n_colors=n_colors
                )
            except Exception as e:
                raise DataValidationError(f"Failed to create design matrix: {e}")

        # Apply linear filter
        generator_signal = design_matrix @ self.filter_true

        # Apply nonlinearity
        try:
            firing_rate = self.nonlinearity_true(generator_signal)
        except Exception as e:
            raise DataValidationError(f"Nonlinearity function failed: {e}")

        # Ensure non-negative firing rates
        firing_rate = np.maximum(firing_rate, 0)

        # Scale firing rate to reasonable values and by bin size
        # Use higher base firing rate to ensure sufficient spikes
        base_firing_rate = 200.0  # Hz - higher rate for better statistics
        expected_spike_count = firing_rate * base_firing_rate * bin_size

        # Add noise by scaling expected spike count
        # Higher noise_level means more variability around expected
        if self.noise_level > 0:
            # Apply multiplicative noise
            noise_factor = 1.0 + self.noise_level * self.rng.randn(len(expected_spike_count))
            noise_factor = np.maximum(noise_factor, 0.1)  # Prevent negative rates
            expected_spike_count = expected_spike_count * noise_factor

        # Ensure non-negative and reasonable spike counts
        expected_spike_count = np.maximum(expected_spike_count, 0)

        # Generate Poisson spikes
        spike_counts = self.rng.poisson(expected_spike_count)

        return spike_counts.astype(np.int32)

    def create_test_dataset(self, duration_minutes: float = 10,
                          bin_size: float = 0.008,
                          **kwargs) -> Dict[str, Any]:
        """
        Complete dataset generation pipeline with metadata.

        Parameters
        ----------
        duration_minutes : float, default=10
            Duration of data in minutes
        bin_size : float, default=0.008
            Temporal bin size in seconds
        **kwargs
            Additional parameters for stimulus generation

        Returns
        -------
        dict
            Complete dataset with metadata
        """
        # Calculate number of time bins
        n_time_bins = int(duration_minutes * 60 / bin_size)

        # Generate stimulus
        stimulus = self.generate_white_noise_stimulus(n_time_bins, **kwargs)

        # Generate responses
        spikes = self.generate_responses(stimulus, bin_size)

        # Create metadata
        metadata = {
            'duration_minutes': duration_minutes,
            'bin_size': bin_size,
            'n_time_bins': n_time_bins,
            'total_spikes': int(np.sum(spikes)),
            'mean_firing_rate_hz': np.sum(spikes) / (n_time_bins * bin_size),
            'filter_true': self.filter_true.copy(),
            'noise_level': self.noise_level,
            'random_seed': self.random_seed,
            'stimulus_shape': stimulus.shape,
            'stimulus_std': np.std(stimulus),
            'stimulus_mean': np.mean(stimulus)
        }

        return {
            'stimulus': stimulus,
            'spikes': spikes,
            'metadata': metadata
        }

    def create_tutorial_dataset(self) -> Dict[str, Any]:
        """
        Generate dataset compatible with MATLAB tutorials for testing.

        Returns
        -------
        dict
            Tutorial-compatible dataset
        """
        # Create a classic center-surround filter
        filter_length = 25
        time_axis = np.arange(filter_length) - filter_length // 2

        # Difference of Gaussians in time
        center = np.exp(-0.5 * (time_axis / 2.0)**2)
        surround = 0.5 * np.exp(-0.5 * (time_axis / 6.0)**2)
        dog_filter = center - surround
        dog_filter = dog_filter / np.linalg.norm(dog_filter)  # Normalize

        # Simple exponential nonlinearity
        def exponential_nonlinearity(x):
            return np.exp(x - np.mean(x))

        # Create generator with tutorial parameters
        tutorial_gen = SyntheticDataGenerator(
            filter_true=dog_filter,
            nonlinearity_true=exponential_nonlinearity,
            noise_level=1.0,
            random_seed=42
        )

        # Generate 10 minutes of data
        dataset = tutorial_gen.create_test_dataset(
            duration_minutes=10,
            bin_size=0.008
        )

        # Add tutorial-specific metadata
        dataset['metadata']['dataset_type'] = 'tutorial'
        dataset['metadata']['filter_type'] = 'difference_of_gaussians'
        dataset['metadata']['nonlinearity_type'] = 'exponential'

        return dataset


def create_separable_spatial_filter(temporal_profile: np.ndarray,
                                   spatial_profile: np.ndarray) -> np.ndarray:
    """
    Create separable spatial-temporal filter.

    Parameters
    ----------
    temporal_profile : np.ndarray
        1D temporal filter profile
    spatial_profile : np.ndarray
        2D spatial filter profile

    Returns
    -------
    np.ndarray
        Separable spatial-temporal filter
    """
    # Outer product to create separable filter
    filter_st = np.outer(temporal_profile, spatial_profile.flatten())
    return filter_st.flatten()


def create_dog_temporal_filter(filter_length: int = 25,
                              center_sigma: float = 2.0,
                              surround_sigma: float = 6.0,
                              surround_weight: float = 0.5) -> np.ndarray:
    """
    Create difference-of-Gaussians temporal filter.

    Parameters
    ----------
    filter_length : int, default=25
        Length of filter in time bins
    center_sigma : float, default=2.0
        Standard deviation of center Gaussian
    surround_sigma : float, default=6.0
        Standard deviation of surround Gaussian
    surround_weight : float, default=0.5
        Relative weight of surround

    Returns
    -------
    np.ndarray
        DoG temporal filter
    """
    time_axis = np.arange(filter_length) - filter_length // 2

    center = np.exp(-0.5 * (time_axis / center_sigma)**2)
    surround = surround_weight * np.exp(-0.5 * (time_axis / surround_sigma)**2)

    dog_filter = center - surround

    # Normalize
    dog_filter = dog_filter / np.linalg.norm(dog_filter)

    return dog_filter


def create_dog_spatial_filter(height: int, width: int,
                             center_sigma: float = 2.0,
                             surround_sigma: float = 6.0,
                             surround_weight: float = 0.5) -> np.ndarray:
    """
    Create difference-of-Gaussians spatial filter.

    Parameters
    ----------
    height : int
        Filter height
    width : int
        Filter width
    center_sigma : float, default=2.0
        Standard deviation of center Gaussian
    surround_sigma : float, default=6.0
        Standard deviation of surround Gaussian
    surround_weight : float, default=0.5
        Relative weight of surround

    Returns
    -------
    np.ndarray
        DoG spatial filter
    """
    y, x = np.ogrid[-height//2:height//2+1, -width//2:width//2+1]

    center = np.exp(-(x**2 + y**2) / (2 * center_sigma**2))
    surround = surround_weight * np.exp(-(x**2 + y**2) / (2 * surround_sigma**2))

    dog_filter = center - surround

    # Normalize
    dog_filter = dog_filter / np.linalg.norm(dog_filter)

    return dog_filter


# Common nonlinearity functions
def exponential_nonlinearity(x: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Exponential nonlinearity: exp(x + offset)"""
    return np.exp(x + offset)


def sigmoid_nonlinearity(x: np.ndarray, gain: float = 1.0,
                        threshold: float = 0.0) -> np.ndarray:
    """Sigmoid nonlinearity: 1 / (1 + exp(-gain * (x - threshold)))"""
    return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))


def rectified_linear_nonlinearity(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Rectified linear nonlinearity: max(0, x - threshold)"""
    return np.maximum(0, x - threshold)


def quadratic_nonlinearity(x: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Quadratic nonlinearity: (x + offset)^2"""
    return (x + offset) ** 2

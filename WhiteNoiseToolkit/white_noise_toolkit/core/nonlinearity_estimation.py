"""
Nonlinearity estimation for the White Noise Analysis Toolkit.

This module implements both parametric and non-parametric methods for
estimating static nonlinearities from generator signals and spike responses.
"""

import warnings
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import scipy.optimize
import scipy.stats
import scipy.interpolate
from ..core.exceptions import (
    NonlinearityFittingError, DataValidationError, 
    InsufficientDataError, NumericalInstabilityError
)


class NonparametricNonlinearity:
    """
    Non-parametric nonlinearity estimation using binning method.
    
    This class estimates the nonlinearity by binning generator signal values
    and computing the mean spike rate in each bin.
    """
    
    def __init__(self, n_bins: int = 25):
        """
        Initialize non-parametric nonlinearity estimator.
        
        Parameters
        ----------
        n_bins : int, default=25
            Number of bins for discretizing generator signal
            
        Raises
        ------
        DataValidationError
            If n_bins is invalid
        """
        if n_bins <= 0:
            raise DataValidationError(f"n_bins must be positive, got {n_bins}")
        
        if n_bins > 1000:
            warnings.warn(
                f"Very large number of bins ({n_bins}) may lead to sparse estimates",
                UserWarning
            )
        
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_centers: Optional[np.ndarray] = None
        self.spike_rates: Optional[np.ndarray] = None
        self.bin_counts: Optional[np.ndarray] = None
        self.fitted = False
        
        # For interpolation
        self._interpolator: Optional[Callable] = None
        self._fit_range: Optional[Tuple[float, float]] = None
    
    def fit(self, generator_signal: np.ndarray, spike_counts: np.ndarray) -> None:
        """
        Fit non-parametric nonlinearity using binning method.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Filter outputs (design_matrix @ filter_weights)
        spike_counts : np.ndarray
            Observed spike counts per time bin
            
        Raises
        ------
        DataValidationError
            If input data is invalid
        InsufficientDataError
            If insufficient data for reliable estimation
        """
        # Validate inputs
        self._validate_inputs(generator_signal, spike_counts)
        
        # Remove invalid values
        valid_mask = np.isfinite(generator_signal) & np.isfinite(spike_counts)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            warnings.warn(f"Removing {n_invalid} invalid data points", UserWarning)
            generator_signal = generator_signal[valid_mask]
            spike_counts = spike_counts[valid_mask]
        
        if len(generator_signal) == 0:
            raise InsufficientDataError(
                "No valid data points after removing invalid values"
            )
        
        # Determine bin edges
        signal_min = np.min(generator_signal)
        signal_max = np.max(generator_signal)
        
        if signal_min == signal_max:
            raise InsufficientDataError(
                "Generator signal has no variance - cannot estimate nonlinearity"
            )
        
        # Add small buffer to edges to include boundary points
        signal_range = signal_max - signal_min
        buffer = signal_range * 0.01  # 1% buffer
        self.bin_edges = np.linspace(
            signal_min - buffer, 
            signal_max + buffer, 
            self.n_bins + 1
        )
        
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Bin the data
        bin_indices = np.digitize(generator_signal, self.bin_edges) - 1
        
        # Ensure indices are within valid range
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        # Compute mean spike rate in each bin
        self.spike_rates = np.zeros(self.n_bins)
        self.bin_counts = np.zeros(self.n_bins, dtype=int)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            self.bin_counts[i] = np.sum(mask)
            
            if self.bin_counts[i] > 0:
                self.spike_rates[i] = np.mean(spike_counts[mask])
            else:
                self.spike_rates[i] = 0
        
        # Handle empty bins
        empty_bins = self.bin_counts == 0
        if empty_bins.any():
            n_empty = empty_bins.sum()
            warnings.warn(
                f"{n_empty} out of {self.n_bins} bins are empty. "
                f"Consider reducing n_bins or using more data.",
                UserWarning
            )
            
            # Interpolate empty bins
            self._interpolate_empty_bins()
        
        # Create interpolation function for continuous evaluation
        self._create_interpolator()
        
        self._fit_range = (signal_min, signal_max)
        self.fitted = True
    
    def _interpolate_empty_bins(self) -> None:
        """Interpolate values for empty bins."""
        if self.spike_rates is None or self.bin_centers is None or self.bin_counts is None:
            return
        
        # Find non-empty bins
        non_empty = self.bin_counts > 0
        
        if non_empty.sum() < 2:
            # Too few non-empty bins for interpolation
            warnings.warn(
                "Too few non-empty bins for interpolation. "
                "Setting empty bins to zero.",
                UserWarning
            )
            return
        
        # Interpolate using non-empty bins
        try:
            interpolator = scipy.interpolate.interp1d(
                self.bin_centers[non_empty],
                self.spike_rates[non_empty],
                kind='linear',
                bounds_error=False,
                fill_value=0
            )
            
            # Fill empty bins
            empty_mask = ~non_empty
            self.spike_rates[empty_mask] = interpolator(self.bin_centers[empty_mask])
            
        except Exception as e:
            warnings.warn(f"Failed to interpolate empty bins: {e}", UserWarning)
    
    def _create_interpolator(self) -> None:
        """Create interpolation function for continuous evaluation."""
        if self.bin_centers is None or self.spike_rates is None:
            return
        
        try:
            self._interpolator = scipy.interpolate.interp1d(
                self.bin_centers,
                self.spike_rates,
                kind='linear',
                bounds_error=False,
                fill_value=0.0  # Use scalar fill value
            )
        except Exception as e:
            warnings.warn(f"Failed to create interpolator: {e}", UserWarning)
    
    def predict(self, generator_signal: np.ndarray) -> np.ndarray:
        """
        Evaluate nonlinearity at given generator signal values.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal values
            
        Returns
        -------
        np.ndarray
            Predicted spike rates
            
        Raises
        ------
        NonlinearityFittingError
            If nonlinearity not fitted
        """
        if not self.fitted:
            raise NonlinearityFittingError("Nonlinearity not fitted. Call fit() first.")
        
        if self._interpolator is None:
            raise NonlinearityFittingError("Interpolator not available")
        
        # Warn if extrapolating beyond fit range
        if self._fit_range is not None:
            signal_min, signal_max = self._fit_range
            out_of_range = (generator_signal < signal_min) | (generator_signal > signal_max)
            if out_of_range.any():
                n_out = out_of_range.sum()
                warnings.warn(
                    f"{n_out} values are outside the fit range "
                    f"[{signal_min:.3f}, {signal_max:.3f}]. "
                    f"Extrapolation may be unreliable.",
                    UserWarning
                )
        
        return self._interpolator(generator_signal)
    
    def get_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return callable function object.
        
        Returns
        -------
        Callable
            Function that evaluates the nonlinearity
        """
        return lambda x: self.predict(x)
    
    def _validate_inputs(self, generator_signal: np.ndarray, 
                        spike_counts: np.ndarray) -> None:
        """
        Validate input data.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal
        spike_counts : np.ndarray
            Spike counts
            
        Raises
        ------
        DataValidationError
            If inputs are invalid
        """
        if generator_signal.shape != spike_counts.shape:
            raise DataValidationError(
                f"Shape mismatch: generator_signal {generator_signal.shape}, "
                f"spike_counts {spike_counts.shape}"
            )
        
        if generator_signal.size == 0:
            raise DataValidationError("Input data is empty")
        
        if np.any(spike_counts < 0):
            warnings.warn("Spike counts contain negative values", UserWarning)
    
    def get_goodness_of_fit(self, generator_signal: np.ndarray, 
                           spike_counts: np.ndarray) -> Dict[str, float]:
        """
        Compute goodness-of-fit metrics.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal values
        spike_counts : np.ndarray
            Observed spike counts
            
        Returns
        -------
        dict
            Goodness-of-fit metrics
        """
        if not self.fitted:
            raise NonlinearityFittingError("Nonlinearity not fitted")
        
        predicted = self.predict(generator_signal)
        
        # R-squared
        ss_res = np.sum((spike_counts - predicted) ** 2)
        ss_tot = np.sum((spike_counts - np.mean(spike_counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Mean squared error
        mse = np.mean((spike_counts - predicted) ** 2)
        
        # Mean absolute error
        mae = np.mean(np.abs(spike_counts - predicted))
        
        # Correlation coefficient
        correlation = np.corrcoef(spike_counts, predicted)[0, 1] if len(spike_counts) > 1 else 0
        
        return {
            'r_squared': r_squared,
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'n_bins': self.n_bins,
            'n_empty_bins': np.sum(self.bin_counts == 0) if self.bin_counts is not None else 0
        }


class ParametricNonlinearity:
    """
    Parametric nonlinearity estimation using maximum likelihood.
    
    This class fits parametric models to the relationship between
    generator signal and spike rates.
    """
    
    def __init__(self, model_type: str = 'exponential'):
        """
        Initialize parametric nonlinearity estimator.
        
        Parameters
        ----------
        model_type : str, default='exponential'
            Type of parametric model:
            - 'exponential': N(g) = a * exp(b * g + c)
            - 'cumulative_normal': N(g) = a * Φ(b * g + c)
            - 'sigmoid': N(g) = a / (1 + exp(-b * (g - c)))
            
        Raises
        ------
        DataValidationError
            If model_type is not supported
        """
        valid_models = ['exponential', 'cumulative_normal', 'sigmoid']
        
        if model_type not in valid_models:
            raise DataValidationError(
                f"Unsupported model_type: {model_type}. "
                f"Supported models: {valid_models}"
            )
        
        self.model_type = model_type
        self.parameters: Optional[np.ndarray] = None
        self.fitted = False
        self.fit_info: Dict[str, Any] = {}
        
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Set up model-specific functions and parameters."""
        if self.model_type == 'exponential':
            self.n_params = 3
            self.param_names = ['a', 'b', 'c']
            self._model_func = self._exponential_model
            self._initial_guess_func = self._exponential_initial_guess
            self._param_bounds = [(1e-6, None), (None, None), (None, None)]
            
        elif self.model_type == 'cumulative_normal':
            self.n_params = 3
            self.param_names = ['a', 'b', 'c']
            self._model_func = self._cumulative_normal_model
            self._initial_guess_func = self._cumulative_normal_initial_guess
            self._param_bounds = [(1e-6, None), (1e-6, None), (None, None)]
            
        elif self.model_type == 'sigmoid':
            self.n_params = 3
            self.param_names = ['a', 'b', 'c']
            self._model_func = self._sigmoid_model
            self._initial_guess_func = self._sigmoid_initial_guess
            self._param_bounds = [(1e-6, None), (1e-6, None), (None, None)]
    
    def _exponential_model(self, generator_signal: np.ndarray, 
                          params: np.ndarray) -> np.ndarray:
        """Exponential model: N(g) = a * exp(b * g + c)"""
        a, b, c = params
        # Clip to prevent overflow
        exponent = np.clip(b * generator_signal + c, -500, 500)
        return a * np.exp(exponent)
    
    def _cumulative_normal_model(self, generator_signal: np.ndarray,
                                params: np.ndarray) -> np.ndarray:
        """Cumulative normal model: N(g) = a * Φ(b * g + c)"""
        a, b, c = params
        return a * scipy.stats.norm.cdf(b * generator_signal + c)
    
    def _sigmoid_model(self, generator_signal: np.ndarray,
                      params: np.ndarray) -> np.ndarray:
        """Sigmoid model: N(g) = a / (1 + exp(-b * (g - c)))"""
        a, b, c = params
        # Clip to prevent overflow
        exponent = np.clip(-b * (generator_signal - c), -500, 500)
        return a / (1 + np.exp(exponent))
    
    def _exponential_initial_guess(self, generator_signal: np.ndarray,
                                  spike_counts: np.ndarray) -> np.ndarray:
        """Initial parameter guess for exponential model."""
        a_init = np.mean(spike_counts)
        b_init = 1.0
        c_init = 0.0
        return np.array([a_init, b_init, c_init])
    
    def _cumulative_normal_initial_guess(self, generator_signal: np.ndarray,
                                        spike_counts: np.ndarray) -> np.ndarray:
        """Initial parameter guess for cumulative normal model."""
        a_init = np.max(spike_counts)
        b_init = 1.0
        c_init = -np.mean(generator_signal)
        return np.array([a_init, b_init, c_init])
    
    def _sigmoid_initial_guess(self, generator_signal: np.ndarray,
                              spike_counts: np.ndarray) -> np.ndarray:
        """Initial parameter guess for sigmoid model."""
        a_init = np.max(spike_counts)
        b_init = 1.0
        c_init = np.median(generator_signal)
        return np.array([a_init, b_init, c_init])
    
    def fit(self, generator_signal: np.ndarray, spike_counts: np.ndarray) -> None:
        """
        Fit parametric nonlinearity using maximum likelihood estimation.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal values
        spike_counts : np.ndarray
            Observed spike counts
            
        Raises
        ------
        NonlinearityFittingError
            If fitting fails
        """
        # Validate inputs
        self._validate_inputs(generator_signal, spike_counts)
        
        # Remove invalid values
        valid_mask = np.isfinite(generator_signal) & np.isfinite(spike_counts)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            warnings.warn(f"Removing {n_invalid} invalid data points", UserWarning)
            generator_signal = generator_signal[valid_mask]
            spike_counts = spike_counts[valid_mask]
        
        if len(generator_signal) < self.n_params:
            raise InsufficientDataError(
                f"Need at least {self.n_params} data points for {self.model_type} model, "
                f"got {len(generator_signal)}"
            )
        
        # Get initial parameter guess
        initial_params = self._initial_guess_func(generator_signal, spike_counts)
        
        # Define objective function (negative log-likelihood for Poisson data)
        def objective(params):
            try:
                predicted_rates = self._model_func(generator_signal, params)
                
                # Ensure positive rates
                predicted_rates = np.maximum(predicted_rates, 1e-10)
                
                # Poisson log-likelihood (negative for minimization)
                log_likelihood = np.sum(
                    spike_counts * np.log(predicted_rates) - predicted_rates
                )
                return -log_likelihood
                
            except (OverflowError, RuntimeWarning):
                return np.inf
        
        # Fit the model
        try:
            result = scipy.optimize.minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=self._param_bounds
            )
            
            if not result.success:
                # Try different optimization method
                result = scipy.optimize.minimize(
                    objective,
                    initial_params,
                    method='Nelder-Mead'
                )
            
            if not result.success:
                raise NonlinearityFittingError(
                    f"Optimization failed: {result.message}",
                    fitting_method='parametric',
                    convergence_info={'success': False, 'message': result.message}
                )
            
            self.parameters = result.x
            self.fit_info = {
                'success': result.success,
                'message': result.message,
                'n_iterations': result.nit,
                'final_cost': result.fun,
                'n_data_points': len(generator_signal)
            }
            
        except Exception as e:
            raise NonlinearityFittingError(
                f"Fitting failed with error: {e}",
                fitting_method='parametric'
            )
        
        self.fitted = True
    
    def predict(self, generator_signal: np.ndarray) -> np.ndarray:
        """
        Evaluate fitted nonlinearity.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal values
            
        Returns
        -------
        np.ndarray
            Predicted spike rates
        """
        if not self.fitted:
            raise NonlinearityFittingError("Nonlinearity not fitted. Call fit() first.")
        
        if self.parameters is None:
            raise NonlinearityFittingError("No fitted parameters available")
        
        return self._model_func(generator_signal, self.parameters)
    
    def get_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return callable function object."""
        return lambda x: self.predict(x)
    
    def get_aic(self, generator_signal: np.ndarray, spike_counts: np.ndarray) -> float:
        """
        Compute Akaike Information Criterion.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal values
        spike_counts : np.ndarray
            Observed spike counts
            
        Returns
        -------
        float
            AIC value
        """
        if not self.fitted:
            raise NonlinearityFittingError("Nonlinearity not fitted")
        
        predicted_rates = self.predict(generator_signal)
        predicted_rates = np.maximum(predicted_rates, 1e-10)
        
        # Poisson log-likelihood
        log_likelihood = np.sum(
            spike_counts * np.log(predicted_rates) - predicted_rates
        )
        
        # AIC = 2k - 2ln(L)
        aic = 2 * self.n_params - 2 * log_likelihood
        
        return aic
    
    def get_bic(self, generator_signal: np.ndarray, spike_counts: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion.
        
        Parameters
        ----------
        generator_signal : np.ndarray
            Generator signal values
        spike_counts : np.ndarray
            Observed spike counts
            
        Returns
        -------
        float
            BIC value
        """
        if not self.fitted:
            raise NonlinearityFittingError("Nonlinearity not fitted")
        
        predicted_rates = self.predict(generator_signal)
        predicted_rates = np.maximum(predicted_rates, 1e-10)
        
        # Poisson log-likelihood
        log_likelihood = np.sum(
            spike_counts * np.log(predicted_rates) - predicted_rates
        )
        
        n = len(generator_signal)
        
        # BIC = ln(n)k - 2ln(L)
        bic = np.log(n) * self.n_params - 2 * log_likelihood
        
        return bic
    
    def _validate_inputs(self, generator_signal: np.ndarray,
                        spike_counts: np.ndarray) -> None:
        """Validate input data."""
        if generator_signal.shape != spike_counts.shape:
            raise DataValidationError(
                f"Shape mismatch: generator_signal {generator_signal.shape}, "
                f"spike_counts {spike_counts.shape}"
            )
        
        if generator_signal.size == 0:
            raise DataValidationError("Input data is empty")
        
        if np.any(spike_counts < 0):
            warnings.warn("Spike counts contain negative values", UserWarning)
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """
        Get fitted parameters as a dictionary.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to values
        """
        if not self.fitted or self.parameters is None:
            return {}
        
        return dict(zip(self.param_names, self.parameters))


def compare_nonlinearity_models(generator_signal: np.ndarray, spike_counts: np.ndarray,
                               models: Optional[list] = None) -> Dict[str, Any]:
    """
    Compare different nonlinearity models using information criteria.
    
    Parameters
    ----------
    generator_signal : np.ndarray
        Generator signal values
    spike_counts : np.ndarray
        Observed spike counts
    models : list, optional
        List of model types to compare. If None, uses all available models.
        
    Returns
    -------
    dict
        Comparison results
    """
    if models is None:
        models = ['exponential', 'cumulative_normal', 'sigmoid', 'nonparametric']
    
    results = {}
    
    for model_type in models:
        try:
            if model_type == 'nonparametric':
                model = NonparametricNonlinearity(n_bins=25)
            else:
                model = ParametricNonlinearity(model_type)
            
            model.fit(generator_signal, spike_counts)
            
            model_results = {
                'fitted': True,
                'model_type': model_type,
            }
            
            if isinstance(model, ParametricNonlinearity):
                model_results['aic'] = model.get_aic(generator_signal, spike_counts)
                model_results['bic'] = model.get_bic(generator_signal, spike_counts)
            
            if isinstance(model, NonparametricNonlinearity):
                gof = model.get_goodness_of_fit(generator_signal, spike_counts)
                model_results.update(gof)
            
            results[model_type] = model_results
            
        except Exception as e:
            results[model_type] = {
                'fitted': False,
                'error': str(e),
                'model_type': model_type
            }
    
    # Determine best model based on AIC (if available)
    fitted_models = {k: v for k, v in results.items() if v.get('fitted', False)}
    
    if fitted_models:
        aic_models = {k: v for k, v in fitted_models.items() if 'aic' in v}
        if aic_models:
            best_model = min(aic_models.keys(), key=lambda k: aic_models[k]['aic'])
            results['best_model_aic'] = best_model
        
        # Best by R-squared
        r2_models = {k: v for k, v in fitted_models.items() if 'r_squared' in v}
        if r2_models:
            best_model_r2 = max(r2_models.keys(), key=lambda k: r2_models[k]['r_squared'])
            results['best_model_r_squared'] = best_model_r2
    
    return results

from abc import ABC, abstractproperty, abstractmethod
import logging
from typing import Callable, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError
from synergy.utils import base as utils

_LOGGER = logging.Logger(__name__)


class DoseResponseModel1D(ABC):
    """-"""

    @abstractmethod
    def fit(self, d, E, **kwargs) -> None:
        """-"""

    @abstractmethod
    def E(self, d):
        """-"""

    @abstractmethod
    def E_inv(self, E):
        """-"""

    @abstractproperty
    def is_specified(self) -> bool:
        """-"""

    @abstractproperty
    def is_fit(self) -> bool:
        """-"""


class ParametricDoseResponseModel1D(DoseResponseModel1D):
    """-"""

    def __init__(self, **kwargs):
        """Ctor."""
        self.fit_function: Callable
        self.jacobian_function: Callable

        self._converged: bool = False
        self._is_fit: bool = False

        self._bounds = self._get_bounds(**kwargs)

        self.sum_of_squares_residuals: Optional[float]
        self.r_squared: Optional[float]
        self.aic: Optional[float]
        self.bic: Optional[float]
        self.bootstrap_parameters = None

    @abstractmethod
    def get_parameters(self) -> list:
        """Returns model parameters"""

    @abstractmethod
    def _set_parameters(self, parameters):
        """Internal method to set model parameters"""

    @abstractmethod
    def E(self, d):
        """Evaluate this model at dose d.

        Parameters
        ----------
        d : array_like
            Doses to calculate effect at

        Returns
        ----------
        effect : array_like
            Evaluate's the model at dose in d
        """

    @abstractmethod
    def E_inv(self, E):
        """Evaluate the inverse of this model.

        Parameters
        ----------
        E : array_like
            Effects to get the doses for

        Returns
        ----------
        doses : array_like
            Doses which achieve effects E using this model. Will return np.nan for effects outside of the model's effect
            range, or for non-invertable models
        """

    @abstractproperty
    def _parameter_names(self) -> list[str]:
        """-"""

    @abstractproperty
    def _default_fit_bounds(self) -> dict[str, tuple[float, float]]:
        """-"""

    def _get_bounds(self, **kwargs):
        """Find all {X}_bounds kwargs and format them into self._bounds as expected by curve_fit()."""
        lower_bounds = []
        upper_bounds = []
        for param in self._parameter_names:
            default_bounds = self._default_fit_bounds.get(param, (-np.inf, np.inf))
            lb, ub = kwargs.pop(f"{param}_bounds", default_bounds)
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        lower_bounds = list(self._transform_params_to_fit(lower_bounds))
        upper_bounds = list(self._transform_params_to_fit(upper_bounds))

        # Log warnings for any other "bounds" passed in
        for key in kwargs:
            if "_bounds" in key:
                _LOGGER.warn(f"Ignoring unexpected bounds for {type(self).__name__}: {key}={kwargs[key]}")
        return lower_bounds, upper_bounds

    def fit(self, d, E, **kwargs):
        """Fit the model to data.

        Parameters
        ----------
        d : array_like
            Array of doses measured

        E : array_like
            Array of effects measured at doses d

        use_jacobian : bool, default=True
            If True, will use the Jacobian to help guide fit. When the number
            of data points is less than a few hundred, this makes the fitting
            slower. However, it also improves the reliability with which a fit
            can be found.

        kwargs
            kwargs to pass to scipy.optimize.curve_fit()
        """
        self._is_fit = True
        d = np.asarray(d)
        E = np.asarray(E)

        # Parse optional kwargs
        use_jacobian = kwargs.pop("use_jacobian", True)
        bootstrap_iterations = kwargs.pop("bootstrap_iterations", 0)
        p0 = kwargs.pop("p0", None)
        if p0 is not None:
            p0 = list(p0)

        # Sanitize initial guesses
        p0 = self._get_initial_guess(d, E, p0, self._bounds)

        # Pass bounds and p0 to kwargs for curve_fit()
        kwargs["p0"] = p0

        with np.errstate(divide="ignore", invalid="ignore"):
            popt = self._fit(d, E, use_jacobian, **kwargs)

        if popt is None:  # curve_fit() failed to fit parameters
            self._converged = False
            return

        # otherwise curve_fit() succeeded
        self._converged = True
        self._set_parameters(popt)

        n_parameters = len(popt)
        n_samples = len(d)
        if n_samples - n_parameters - 1 > 0:  # TODO: What is this watching out for?
            self._score(d, E)
            kwargs["p0"] = self._transform_params_to_fit(popt)
            self._bootstrap_resample(
                d,
                E,
                use_jacobian,
                bootstrap_iterations,
                **kwargs,
            )

    def get_confidence_intervals(self, confidence_interval: float = 95):
        """Returns the lower bound and upper bound estimate for each parameter.

        Parameters:
        -----------
        confidence_interval : float, default=95
            % confidence interval to return. Must be between 0 and 100.
        """
        if not self.is_specified:
            raise ModelNotParameterizedError()
        if not self.is_fit:
            raise ModelNotFitToDataError()
        if confidence_interval < 0 or confidence_interval > 100:
            raise ValueError(f"confidence_interval must be between 0 and 100 ({confidence_interval})")
        if self.bootstrap_parameters is None:
            raise ValueError(
                "Model must have been fit with bootstrap_iterations > 0 to get parameter confidence intervals"
            )

        lb = (100 - confidence_interval) / 2.0
        ub = 100 - lb
        return np.percentile(self.bootstrap_parameters, [lb, ub], axis=0).transpose()

    def _get_initial_guess(self, d, E, p0, bounds):
        """Transform user supplied initial guess to correct scale, and/or guess p0."""
        if p0:
            p0 = list(self._transform_params_to_fit(p0))
        return utils.sanitize_initial_guess(p0, bounds)

    def _transform_params_from_fit(self, params):
        """Transform parameters from curve-fitting scale to linear scale.

        For instance, models that fit logh and logC must transform those to h and C
        """
        return params

    def _transform_params_to_fit(self, params):
        """Transform parameters to scale used for curve fitting.

        For instance, models that fit logh and logC must transform from h and C
        """
        return params

    def _fit(self, d, E, use_jacobian: bool, **kwargs):
        """Fit the model to data (d, E)"""
        jac = self.jacobian_function if use_jacobian and self.jacobian_function is not None else None
        popt = curve_fit(
            self.fit_function,
            d,
            E,
            bounds=self._bounds,
            jac=jac,
            **kwargs,
        )[0]

        if True in np.isnan(popt):
            return None
        return self._transform_params_from_fit(popt)

    def _score(self, d, E):
        """Calculate goodness of fit and model quality scores

        This calculations
        - `sum_of_squares_residuals`
        - `r_squared`
        - `aic` (Akaike Information Criterion)
        - `bic` (Bayesian Information Criterion)

        Called automatically during model.fit(d1, d2, E)

        :param ArrayLike d: Doses
        :param ArrayLike E: Measured dose-response effect at doses d
        """
        if not (self.is_specified and self.is_fit):
            raise ModelNotFitToDataError("Must fit the model to data before scoring")

        n_parameters = len(self.get_parameters())
        n_datapoints = len(E)

        self.sum_of_squares_residuals = utils.residual_ss_1d(d, E, self.E)
        self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
        self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, n_datapoints)
        self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, n_datapoints)

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, **kwargs):
        """Identify confidence intervals for parameters using bootstrap resampling.

        Residuals are randomly sampled from a normal distribution with :math:`\sigma = \sqrt{\frac{RSS}{n - N}}`
        where :math:`RSS` is the residual sum of square, :math:`n` is the number of data points, and :math:`N` is the
        number of parameters.
        """
        if not self.is_specified:
            raise ModelNotParameterizedError()
        if not self.is_converged:
            raise ModelNotFitToDataError()

        n_data_points = len(E)
        n_parameters = len(self.get_parameters())

        sigma_residuals = np.sqrt(self.sum_of_squares_residuals / (n_data_points - n_parameters))  # type: ignore

        E_model = self.E(d)
        bootstrap_parameters = []

        for _ in range(bootstrap_iterations):
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide="ignore", invalid="ignore"):
                popt1 = self._fit(d, E_iteration, use_jacobian=use_jacobian, **kwargs)

            if popt1 is not None:
                bootstrap_parameters.append(popt1)

        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

    @property
    def is_specified(self) -> bool:
        """True if all parameters are set."""
        parameters = self.get_parameters()
        if parameters is None:
            return False

        return None not in parameters and not np.isnan(np.asarray(parameters)).any()

    @property
    def is_converged(self) -> bool:
        """-"""
        return self._converged

    @property
    def is_fit(self) -> bool:
        """-"""
        return self._is_fit

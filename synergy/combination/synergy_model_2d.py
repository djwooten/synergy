"""Base classes for 2-drug synergy models."""

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Callable, Optional
import logging

from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np

from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.utils import base as utils

_LOGGER = logging.Logger(__name__)


class SynergyModel2D(ABC):
    """-"""

    def __init__(
        self, drug1_model: Optional[DoseResponseModel1D] = None, drug2_model: Optional[DoseResponseModel1D] = None
    ):
        """-"""
        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class

        drug1_model = deepcopy(drug1_model)
        drug2_model = deepcopy(drug2_model)

        self.drug1_model: DoseResponseModel1D = utils.sanitize_single_drug_model(
            drug1_model, default_type, required_type, **self._default_drug1_kwargs
        )
        self.drug2_model: DoseResponseModel1D = utils.sanitize_single_drug_model(
            drug2_model, default_type, required_type, **self._default_drug2_kwargs
        )

    @abstractmethod
    def fit(self, d1, d2, E, **kwargs):
        """-"""

    @abstractmethod
    def E_reference(self, d1, d2):
        """-"""

    @abstractproperty
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""

    @abstractproperty
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""

    @abstractproperty
    def is_specified(self):
        """-"""

    @abstractproperty
    def is_fit(self):
        """-"""

    @property
    def _default_drug1_kwargs(self) -> dict:
        """-"""
        return {}

    @property
    def _default_drug2_kwargs(self) -> dict:
        """-"""
        return {}


class DoseDependentSynergyModel2D(SynergyModel2D):
    """-"""

    def __init__(
        self, drug1_model: Optional[DoseResponseModel1D] = None, drug2_model: Optional[DoseResponseModel1D] = None
    ):
        """-"""
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model)
        self.synergy = None
        self.d1 = None
        self.d2 = None
        self.reference = None
        self._is_fit = False

    def fit(self, d1, d2, E, **kwargs):
        """-"""
        self.d1 = d1
        self.d2 = d2
        self.synergy = d1 * np.nan
        self._is_fit = True

        # Fit the single drug models if they were not pre-fit by the user
        if not self.drug1_model.is_specified:
            mask = np.where(d2 == min(d2))
            self.drug1_model.fit(d1[mask], E[mask], **kwargs)

        if not self.drug2_model.is_specified:
            mask = np.where(d1 == min(d1))
            self.drug2_model.fit(d2[mask], E[mask], **kwargs)

        if not self.is_specified:
            raise ModelNotParameterizedError("Cannot calculate synergy because the model is not specified")

        self.reference = self.E_reference(d1, d2)
        self.synergy = self._get_synergy(d1, d2, E)

        return self.synergy

    @property
    def is_specified(self):
        """-"""
        return self.drug1_model is not None and self.drug2_model is not None

    @property
    def is_fit(self):
        """-"""
        return self._is_fit

    @abstractmethod
    def _get_synergy(self, d1, d2, E):
        """-"""

    def _sanitize_synergy(self, d1, d2, synergy, default_val: float):
        if hasattr(synergy, "__iter__"):
            synergy[(d1 == 0) | (d2 == 0)] = default_val
        elif d1 == 0 or d2 == 0:
            synergy = default_val
        return synergy


class ParametricSynergyModel2D(SynergyModel2D):
    """-"""

    def __init__(
        self,
        drug1_model: Optional[DoseResponseModel1D] = None,
        drug2_model: Optional[DoseResponseModel1D] = None,
        **kwargs,
    ):
        """Ctor."""
        self._bounds = self._get_bounds(**kwargs)
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model)
        self.fit_function: Callable
        self.jacobian_function: Callable

        self._converged: bool = False
        self._is_fit: bool = False

        self.sum_of_squares_residuals: Optional[float]
        self.r_squared: Optional[float]
        self.aic: Optional[float]
        self.bic: Optional[float]
        self.bootstrap_parameters = None

    @abstractmethod
    def E(self, d1, d2):
        """-"""

    @abstractmethod
    def get_parameters(self):
        """-"""

    @abstractmethod
    def _set_parameters(self, parameters):
        """-"""

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

    def fit(self, d1, d2, E, **kwargs):
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
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)

        # Parse optional kwargs
        use_jacobian = kwargs.pop("use_jacobian", True)
        bootstrap_iterations = kwargs.pop("bootstrap_iterations", 0)
        p0 = kwargs.pop("p0", None)
        if p0 is not None:
            p0 = list(p0)

        # Sanitize initial guesses
        p0 = self._get_initial_guess(d1, d2, E, p0)

        # Pass bounds and p0 to kwargs for curve_fit()
        kwargs["p0"] = p0

        with np.errstate(divide="ignore", invalid="ignore"):
            popt = self._fit(d1, d2, E, use_jacobian, **kwargs)

        if popt is None:  # curve_fit() failed to fit parameters
            self._converged = False
            return

        # otherwise curve_fit() succeeded
        self._converged = True
        self._set_parameters(popt)

        n_parameters = len(popt)
        n_samples = len(d1)
        if n_samples - n_parameters - 1 > 0:  # TODO: What is this watching out for?
            self._score(d1, d2, E)
            kwargs["p0"] = self._transform_params_to_fit(popt)
            self._bootstrap_resample(d1, d2, E, use_jacobian, bootstrap_iterations, **kwargs)

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

    def _get_initial_guess(self, d1, d2, E, p0):
        """Transform user supplied initial guess to correct scale, and/or guess p0."""
        if p0:
            p0 = list(self._transform_params_to_fit(p0))
        return utils.sanitize_initial_guess(p0, self._bounds)

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

    def _fit(self, d1, d2, E, use_jacobian: bool, **kwargs):
        """Fit the model to data (d, E)"""
        jac = self.jacobian_function if use_jacobian else None
        if use_jacobian and jac is None:
            _LOGGER.warn(f"No jacobian function is specified for {type(self).__name__}, ignoring `use_jacobian`.")
        popt = curve_fit(
            self.fit_function,
            (d1, d2),
            E,
            bounds=self._bounds,
            jac=jac,
            **kwargs,
        )[0]

        if np.isnan(popt).any():
            return None
        return self._transform_params_from_fit(popt)

    def _score(self, d1, d2, E):
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

        self.sum_of_squares_residuals = utils.residual_ss(d1, d2, E, self.E)
        self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
        self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, n_datapoints)
        self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, n_datapoints)

    def _bootstrap_resample(self, d1, d2, E, use_jacobian, bootstrap_iterations, **kwargs):
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

        E_model = self.E(d1, d2)
        bootstrap_parameters = []

        for _ in range(bootstrap_iterations):
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide="ignore", invalid="ignore"):
                popt1 = self._fit(d1, d2, E_iteration, use_jacobian=use_jacobian, **kwargs)

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

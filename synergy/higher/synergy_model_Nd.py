"""Base classes for N-drug synergy models (N > 2)."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Optional
import logging

from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np

from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.utils import base as utils
from synergy.utils.model_mixins import ParametricModelMixins

_LOGGER = logging.Logger(__name__)


class SynergyModelND(ABC):
    """Base class for all N-drug synergy models (N > 2)."""

    def __init__(self, single_drug_models: Optional[list[DoseResponseModel1D]] = None):
        """-"""
        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class
        if single_drug_models:
            if len(single_drug_models) < 2:
                raise ValueError(f"Cannot fit a model with fewer than two drugs (N={len(single_drug_models)})")

            self.N = len(self.single_drug_models)
            self.single_drug_models = [
                utils.sanitize_single_drug_model(
                    deepcopy(model), default_type, required_type, **self._default_single_drug_kwargs
                )
                for model in single_drug_models
            ]
        else:
            self.N = -1
            self.single_drug_models = None

    @abstractmethod
    def fit(self, d, E, **kwargs):
        """-"""

    def _fit_single_drugs(self, d, E, **kwargs):
        """-"""
        N = d.shape[1]

        if self.N > 1 and N != self.N:
            raise ValueError(f"This is an {self.N} drug model, which cannot be used with {N}-dimensional dose data")
        if self.single_drug_models is None:
            self.single_drug_models = []

        # Fit all non-specified single drug models
        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class
        for single_idx in range(N):
            if single_idx >= len(self.single_drug_models):
                model = utils.sanitize_single_drug_model(
                    default_type, default_type, required_type, **self._default_single_drug_kwargs
                )
                self.single_drug_models.append(model)
            model = self.single_drug_models[single_idx]
            if model.is_specified:
                continue
            mask = self._get_drug_alone_mask(d, single_idx)
            single_kwargs = deepcopy(kwargs)
            single_kwargs.pop("bootstrap_iterations")
            # TODO: Get single drug p0
            model.fit(d[mask, single_idx].flatten(), E[mask], **single_kwargs)

    def _get_drug_alone_mask(self, d, drug_idx):
        """Find all dose combinations where only one drug is present.

        Note: other drugs are considered to be absent as long as they are at their minimum dose.
        """
        N = d.shape[1]
        mask = d[:, drug_idx] >= 0  # This inits it to "True"
        for other_idx in range(N):
            if other_idx == drug_idx:
                continue
            mask == mask & (d[:, other_idx] == np.min(d[:, other_idx]))
        return np.where(mask)

    @abstractmethod
    def E_reference(self, d):
        """-"""

    @property
    @abstractmethod
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""

    @property
    @abstractmethod
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""

    @property
    @abstractmethod
    def is_specified(self):
        """-"""

    @property
    @abstractmethod
    def is_fit(self):
        """-"""

    @property
    def _default_single_drug_kwargs(self) -> dict:
        """Default keyword arguments for all single drug models.

        Used if the user does not specify any single drug models.
        """
        return {}


class DoseDependentSynergyModelND(SynergyModelND):
    """Base class for N-drug synergy models (N > 2) for which synergy varies based on dose."""

    def __init__(self, single_drug_models: Optional[list[DoseResponseModel1D]] = None):
        """-"""
        super().__init__(single_drug_models=single_drug_models)
        self.synergy = None
        self.d = None
        self.reference = None
        self._is_fit = False

    def fit(self, d, E, **kwargs):
        """-"""
        self.d = d
        self.synergy = E * np.nan

        self._fit_single_drugs(d, E, **kwargs)
        if not self.is_specified:
            raise ModelNotParameterizedError("The model failed to fit")

        self._is_fit = True
        self.reference = self.E_reference(d)
        self.synergy = self._get_synergy(d, E)

        return self.synergy

    @property
    def is_specified(self):
        """True if all single drug models are specified."""
        if not self.single_drug_models or len(self.single_drug_models) < 2:
            return False
        for model in self.single_drug_models:
            if not model.is_specified:
                return False
        return True

    @property
    def is_fit(self):
        """True if the model was fit to data."""
        return self._is_fit

    @abstractmethod
    def _get_synergy(self, d, E):
        """Return the synergy for the given dose combination(s)."""

    def _is_monotherapy(doses):
        """Return True if no more than 1 drug is present."""
        vals, counts = np.unique(doses > 0, return_counts=True)
        drugs_present_count = counts[vals]
        if len(drugs_present_count) == 0:
            return True
        return drugs_present_count[0] == 1

    def _get_single_drug_mask(self, d):
        """Return a mask of where no more than 1 drug is present.

        This helps to set synergy to the default value for monotherapy combinations.
        """
        return np.where(np.apply_along_axis(self._is_monotherapy, 1, d))

    def _sanitize_synergy(self, d, synergy, default_val: float):
        if len(d.shape) == 2:
            synergy[self._get_single_drug_mask(d)] = default_val
        elif len(d.shape) == 1:
            if self._is_monotherapy(d):
                synergy = default_val
        else:
            raise ValueError("d must be a 1 or 2 dimensional array")
        return synergy


class ParametricSynergyModelND(SynergyModelND):
    """-"""

    def __init__(
        self,
        single_drug_models: Optional[list[DoseResponseModel1D]] = None,
        **kwargs,
    ):
        """Ctor."""
        self._set_init_parameters(**kwargs)
        self._bounds = self._get_bounds(**kwargs)
        super().__init__(single_drug_models=single_drug_models)
        self.fit_function: Callable
        self.jacobian_function: Callable  # Currently no jacobian for any N-drug models

        self._converged: bool = False
        self._is_fit: bool = False

        self.sum_of_squares_residuals: Optional[float]
        self.r_squared: Optional[float]
        self.aic: Optional[float]
        self.bic: Optional[float]
        self.bootstrap_parameters = None

    @abstractmethod
    def E(self, d):
        """-"""

    def get_parameters(self) -> dict[str, Any]:
        """Returns model's parameters"""
        return {param: self.__getattribute__(param) for param in self._parameter_names}

    def _set_init_parameters(self, **kwargs):
        """-"""
        for param in self._parameter_names:
            self.__setattr__(param, kwargs.get(param, None))

    @abstractmethod
    def _set_parameters(self, parameters):
        """-"""

    @property
    @abstractmethod
    def _parameter_names(self) -> list[str]:
        """-"""

    @property
    @abstractmethod
    def _default_fit_bounds(self) -> dict[str, tuple[float, float]]:
        """-"""

    def _get_bounds(self, **kwargs):
        """Find all {X}_bounds kwargs and format them into self._bounds as expected by curve_fit()."""
        lower_bounds = []
        upper_bounds = []
        default_bounds = self._default_fit_bounds
        for param in self._parameter_names:
            lb, ub = kwargs.pop(f"{param}_bounds", default_bounds.get(param, (-np.inf, np.inf)))
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        lower_bounds = list(self._transform_params_to_fit(lower_bounds))
        upper_bounds = list(self._transform_params_to_fit(upper_bounds))

        # Log warnings for any other "bounds" passed in
        for key in kwargs:
            if "_bounds" in key:
                _LOGGER.warning(f"Ignoring unexpected bounds for {type(self).__name__}: {key}={kwargs[key]}")
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
        intervals = np.percentile(self.bootstrap_parameters, [lb, ub], axis=0).transpose()
        return dict(zip(self._parameter_names, intervals))

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
            _LOGGER.warning(f"No jacobian function is specified for {type(self).__name__}, ignoring `use_jacobian`.")
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

    def _make_summary_row(
        self,
        key: str,
        comp_val: int,
        val: float,
        ci: dict[str, tuple[float, float]],
        tol: float,
        log: bool,
        gt_outcome: str,
        lt_outcome: str,
        default_outcome: str = "additive",
    ):
        if ci:
            lb, ub = ci[key]
            if lb > comp_val:
                comparison = f"> {comp_val}"
                outcome = gt_outcome
            elif ub < comp_val:
                comparison = f"< {comp_val}"
                outcome = lt_outcome
            else:
                comparison = f"~= {comp_val}"
                outcome = default_outcome
            return [key, f"{val:0.3g}", f"({lb:0.3g}, {ub:0.3g})", comparison, outcome]
        val_scaled = np.log(val) if log else val
        comp_val_scaled = np.log(comp_val) if log else comp_val
        if val_scaled > comp_val_scaled + tol:
            comparison = f"> {comp_val}"
            outcome = gt_outcome
        elif val_scaled < comp_val_scaled - tol:
            comparison = f"< {comp_val}"
            outcome = lt_outcome
        else:
            comparison = f"~= {comp_val}"
            outcome = default_outcome
        return [key, f"{val:0.3g}", comparison, outcome]

    @property
    def is_specified(self) -> bool:
        """True if all parameters are set."""
        parameters = list(self.get_parameters().values())

        return None not in parameters and not np.isnan(np.asarray(parameters)).any()

    @property
    def is_converged(self) -> bool:
        """-"""
        return self._converged

    @property
    def is_fit(self) -> bool:
        """-"""
        return self._is_fit

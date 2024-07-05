"""Base classes for N-drug synergy models (N > 2)."""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Union


import numpy as np
from numpy import typing as npt
from scipy.optimize import curve_fit


from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.typing import DRModel1D
from synergy import utils
from synergy.utils import dose_utils
from synergy.utils.model_mixins import ParametricModelMixins

_LOGGER = logging.Logger(__name__)


class SynergyModelND(ABC):
    """Base class for all N-drug synergy models (N > 2)."""

    def __init__(self, single_drug_models: Optional[Sequence[DRModel1D]] = None):
        """-"""
        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class

        self.single_drug_models: Optional[list[DoseResponseModel1D]] = None
        if not hasattr(self, "N"):
            self.N = -1

        if single_drug_models:
            if len(single_drug_models) < 2:
                raise ValueError(f"Cannot fit a model with fewer than two drugs (N={len(single_drug_models)})")

            self.single_drug_models = [
                utils.sanitize_single_drug_model(
                    deepcopy(model), default_type, required_type, **self._get_default_single_drug_kwargs(idx)
                )
                for idx, model in enumerate(single_drug_models)
            ]
            self.N = len(self.single_drug_models)

    @abstractmethod
    def fit(self, d: npt.ArrayLike, E: npt.ArrayLike, **kwargs):
        """-"""

    def _fit_single_drugs(self, d: npt.ArrayLike, E: npt.ArrayLike, **kwargs):
        """-"""
        N = d.shape[-1]

        if self.N > 1 and N != self.N:
            raise ValueError(f"This is an {self.N} drug model, which cannot be used with {N}-dimensional dose data")

        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class

        if self.single_drug_models is None:
            self.single_drug_models = [default_type] * N
            self.N = N

        # Fit all non-specified single drug models
        model: Union[DoseResponseModel1D, type[DoseResponseModel1D]]
        for single_idx, model in enumerate(self.single_drug_models):
            model = utils.sanitize_single_drug_model(
                model, default_type, required_type, **self._get_default_single_drug_kwargs(single_idx)
            )
            self.single_drug_models[single_idx] = model
            if model.is_specified:
                continue
            mask = dose_utils.get_drug_alone_mask_ND(d, single_idx)
            single_kwargs = deepcopy(kwargs)
            single_kwargs.pop("bootstrap_iterations", None)
            # TODO: Get single drug p0
            model.fit(d[mask, single_idx].flatten(), E[mask], **single_kwargs)

    @abstractmethod
    def E_reference(self, d: npt.ArrayLike):
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

    def _get_default_single_drug_kwargs(self, drug_idx: int) -> dict[str, Any]:
        """Default keyword arguments for single drug models.

        This is used for each single drug unless an already instantiated version is provided in __init__().
        """
        return {}


class DoseDependentSynergyModelND(SynergyModelND):
    """Base class for N-drug synergy models (N > 2) for which synergy varies based on dose."""

    def __init__(self, single_drug_models: Sequence[DoseResponseModel1D] = None):
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

    def _sanitize_synergy(self, d, synergy, default_val: float):
        """Replace non-combinations with default synergy value."""
        if len(d.shape) == 2:
            synergy[dose_utils.get_monotherapy_mask_ND(d)] = default_val
        elif len(d.shape) == 1:
            if dose_utils.is_monotherapy(d):
                synergy = default_val
        else:
            raise ValueError("d must be a 1 or 2 dimensional array")
        return synergy


class ParametricSynergyModelND(SynergyModelND):
    """-"""

    def __init__(
        self,
        single_drug_models: Optional[Sequence[DoseResponseModel1D]] = None,
        num_drugs: int = -1,
        **kwargs,
    ):
        """Ctor."""
        if single_drug_models:
            self.N = len(single_drug_models)
        else:
            self.N = num_drugs
        ParametricModelMixins.set_init_parameters(self, self._parameter_names, **kwargs)
        ParametricModelMixins.set_bounds(
            self, self._transform_params_to_fit, self._default_fit_bounds, self._parameter_names, **kwargs
        )
        super().__init__(single_drug_models=single_drug_models)

        self.fit_function: Callable
        self.jacobian_function: Optional[Callable] = None  # Currently no jacobian for any N-drug models

        self._converged: bool = False
        self._is_fit: bool = False

        self.sum_of_squares_residuals: Optional[float]
        self.r_squared: Optional[float]
        self.aic: Optional[float]
        self.bic: Optional[float]
        self.bootstrap_parameters = None

    def E(self, d):
        """-"""
        if len(d.shape) == 0 or len(d.shape) > 2:
            raise ValueError("d must be an array with columns representing each drug and rows each dose")

        n = d.shape[-1]
        if n != self.N:
            raise ValueError(f"Expected d to have {self.N} columns, but got {n}")

        if not self.is_specified:
            return ModelNotParameterizedError()

        params = self._transform_params_to_fit(self._get_parameters())
        return self.fit_function(d, *params)

    def get_parameters(self) -> dict[str, Any]:
        """Returns model's parameters"""
        return {param: self.__getattribute__(param) for param in self._parameter_names}

    @property
    @abstractmethod
    def _parameter_names(self) -> list[str]:
        """-"""

    @property
    @abstractmethod
    def _default_fit_bounds(self) -> dict[str, tuple[float, float]]:
        """-"""

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
        max_iterations = kwargs.pop("max_iterations", 10000)
        p0 = kwargs.pop("p0", None)
        if p0 is not None:
            p0 = list(p0)

        self._fit_single_drugs(d, E)

        # Sanitize initial guesses
        with np.errstate(divide="ignore", invalid="ignore"):
            p0 = self._get_initial_guess(d, E, p0)

        # Pass bounds and p0 to kwargs for curve_fit()
        kwargs["p0"] = p0

        with np.errstate(divide="ignore", invalid="ignore"):
            popt = self._fit(d, E, use_jacobian, **kwargs)

        if popt is None:  # curve_fit() failed to fit parameters
            self._converged = False
            return

        # otherwise curve_fit() succeeded
        self._converged = True
        ParametricModelMixins.set_parameters(self, self._parameter_names, *popt)

        n_parameters = len(popt)
        if len(d.shape) == 1:
            n_samples = 1
        else:
            n_samples = d.shape[0]
        if n_samples - n_parameters - 1 > 0:  # TODO: What is this watching out for?
            self._score(d, E)
            kwargs["p0"] = self._transform_params_to_fit(popt)
            ParametricModelMixins.bootstrap_parameter_ranges(
                self, E, use_jacobian, bootstrap_iterations, max_iterations, d, **kwargs
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
        intervals = np.percentile(self.bootstrap_parameters, [lb, ub], axis=0).transpose()
        return dict(zip(self._parameter_names, intervals))

    def _get_initial_guess(self, d, E, p0):
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

    def _get_parameters(self):
        """-"""
        return [self.__getattribute__(param) for param in self._parameter_names]

    def _fit(self, d, E, use_jacobian: bool, **kwargs):
        """Fit the model to data (d, E)"""
        jac = self.jacobian_function if use_jacobian else None
        if use_jacobian and jac is None:
            _LOGGER.warning(f"No jacobian function is specified for {type(self).__name__}, ignoring `use_jacobian`.")
        popt = curve_fit(
            self.fit_function,
            d,
            E,
            bounds=self._bounds,
            jac=jac,
            **kwargs,
        )[0]

        if np.isnan(popt).any():
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

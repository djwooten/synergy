"""Base classes for 2-drug synergy models."""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from scipy.optimize import curve_fit

from synergy import utils
from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.utils.model_mixins import ParametricModelMixins

_LOGGER = logging.Logger(__name__)


class SynergyModel2D(ABC):
    """Base class for all 2-drug synergy models."""

    def __init__(
        self,
        drug1_model: Optional[DoseResponseModel1D] = None,
        drug2_model: Optional[DoseResponseModel1D] = None,
        **kwargs,
    ):
        """Ctor."""
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
        """Fit the model to data.

        :param ArrayLike d1: Concentration of drug 1
        :param ArrayLike d2: Concentration of drug 2
        :param ArrayLike E: Effect of the combination of drugs at doses d1 and d2
        :param dict kwargs: Additional keyword arguments for fitting
        """

    @abstractmethod
    def E_reference(self, d1, d2):
        """Return the expected effect of the combination of drugs at doses d1 and d2.

        :param ArrayLike d1: Concentration of drug 1
        :param ArrayLike d2: Concentration of drug 2
        :return ArrayLike: Reference (additive) values of E at the given doses
        """

    @property
    @abstractmethod
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        """The required type of single drug model for this synergy model."""

    @property
    @abstractmethod
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        """The default type of single drug model for this synergy model."""

    @property
    @abstractmethod
    def is_specified(self):
        """True if all parameters are set."""

    @property
    @abstractmethod
    def is_fit(self):
        """True if the model has been fit to data."""

    @property
    def _default_drug1_kwargs(self) -> dict:
        """Default keyword arguments for drug 1's model.

        Used if drug1_model is None.
        """
        return {}

    @property
    def _default_drug2_kwargs(self) -> dict:
        """Default keyword arguments for drug 2's model.

        Used if drug2_model is None.
        """
        return {}


class DoseDependentSynergyModel2D(SynergyModel2D):
    """Base class for 2-drug synergy models for which synergy varies based on dose."""

    def __init__(self, drug1_model=None, drug2_model=None):
        """Ctor."""
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model)
        self.synergy = None
        self.d1 = None
        self.d2 = None
        self.reference = None
        self._is_fit = False

    def fit(self, d1, d2, E, **kwargs):
        """Fit the model to data.

        :param ArrayLike d1: Concentration of drug 1
        :param ArrayLike d2: Concentration of drug 2
        :param ArrayLike E: Effect of the combination of drugs at doses d1 and d2
        :param dict kwargs:
            - use_jacobian: whether to use the model jacobian when fitting single-drug models
            - Additional keyword arguments for ``scipy.optimize.curve_fit()``
        :return ArrayLike: Synergy values
        """
        self.d1 = d1
        self.d2 = d2
        self.synergy = d1 * np.nan

        # Fit the single drug models if they were not pre-fit by the user
        if not self.drug1_model.is_specified:
            mask = np.where(d2 == min(d2))
            self.drug1_model.fit(d1[mask], E[mask], **kwargs)

        if not self.drug2_model.is_specified:
            mask = np.where(d1 == min(d1))
            self.drug2_model.fit(d2[mask], E[mask], **kwargs)

        if not self.is_specified:
            raise ModelNotParameterizedError("The model failed to fit")

        self._is_fit = True
        self.reference = self.E_reference(d1, d2)
        self.synergy = self._get_synergy(d1, d2, E)

        return self.synergy

    @property
    def is_specified(self):
        """True if each single-drug model is specified."""
        return (
            self.drug1_model is not None
            and self.drug2_model is not None
            and self.drug1_model.is_specified
            and self.drug2_model.is_specified
        )

    @property
    def is_fit(self):
        """True if the model has been fit to data."""
        return self._is_fit

    @abstractmethod
    def _get_synergy(self, d1, d2, E):
        """Calculate synergy at doses d1 and d2."""

    def _sanitize_synergy(self, d1, d2, synergy, default_val: float):
        """Set the synergy to the default value when one of the doses is 0."""
        if hasattr(synergy, "__iter__"):
            synergy[(d1 == 0) | (d2 == 0)] = default_val
        elif d1 == 0 or d2 == 0:
            synergy = default_val
        return synergy


class ParametricSynergyModel2D(SynergyModel2D):
    """Base class for parametric 2-drug synergy models."""

    def __init__(self, drug1_model=None, drug2_model=None, **kwargs):
        """Ctor."""
        self._bounds: Tuple[Sequence[float], Sequence[float]]
        ParametricModelMixins.set_init_parameters(self, self._parameter_names, **kwargs)
        ParametricModelMixins.set_bounds(
            self, self._transform_params_to_fit, self._default_fit_bounds, self._parameter_names, **kwargs
        )
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
        """Calculate the expected effect of the combination of drugs at doses d1 and d2.

        :param ArrayLike d1: Concentration of drug 1
        :param ArrayLike d2: Concentration of drug 2
        :return ArrayLike: Expected effect of the combination of drugs at doses d1 and d2
        """

    def get_parameters(self) -> Dict[str, Any]:
        """Return the model's parameters as a dict keyed by parameter name.

        :return Dict[str, Any]: Model parameters
        """
        return {param: self.__getattribute__(param) for param in self._parameter_names}

    @abstractmethod
    def _set_parameters(self, parameters):
        """Set the model parameters."""

    @property
    @abstractmethod
    def _parameter_names(self) -> List[str]:
        """String names of each parameter."""

    @property
    @abstractmethod
    def _default_fit_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Default bounds for each parameter."""

    def fit(self, d1, d2, E, **kwargs):
        """Fit the model to data.

        :param ArrayLike d1: Concentration of drug 1
        :param ArrayLike d2: Concentration of drug 2
        :param ArrayLike E: Effect of the combination of drugs at doses d1 and d2
        :param dict kwargs:
            - p0: Initial parameter guesses
            - bootstrap_iterations: Number of bootstrap iterations to perform to estimate confidence intervals
            - use_jacobian: whether to use the model jacobian when fitting
            - Additional kwargs for ``scipy.optimize.curve_fit()``
        """
        self._is_fit = True
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)

        # Parse optional kwargs
        use_jacobian = kwargs.pop("use_jacobian", True if self.jacobian_function is not None else False)
        bootstrap_iterations = kwargs.pop("bootstrap_iterations", 0)
        max_iterations = kwargs.pop("max_iterations", 10000)
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
            ParametricModelMixins.bootstrap_parameter_ranges(
                self, E, use_jacobian, bootstrap_iterations, max_iterations, d1, d2, **kwargs
            )

    def get_confidence_intervals(self, confidence_interval: float = 95) -> Dict[str, Tuple[float, float]]:
        """Return the lower bound and upper bound estimates for each parameter.

        Parameters
        ----------
        confidence_interval : float, default=95
            % confidence interval to return. Must be between 0 and 100.

        Returns
        -------
        Dict[str, Tuple[float, float]]
            Lower and upper bounds for each parameter keyed by parameter name.
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

    @property
    def is_specified(self) -> bool:
        """True if all parameters are set."""
        parameters = list(self.get_parameters().values())

        return None not in parameters and not np.isnan(np.asarray(parameters)).any()

    @property
    def is_converged(self) -> bool:
        """True if model.fit() was called and the optimization converged."""
        return self._converged

    @property
    def is_fit(self) -> bool:
        """True if model.fit() was called."""
        return self._is_fit

    @abstractmethod
    def summarize(self, confidence_interval: float = 95, tol: float = 0.01):
        """Print a summary table of the synergy model.

        :param float confidence_interval: The confidence interval to use for parameter estimates (must be between 0 and
            100).
        :param float tol: The tolerance around additivity for determining synergism or antagonism.
        """

#    Copyright (C) 2020 David J. Wooten
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

from synergy.utils import base as utils
from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError


class ParameterizedModel1D(ABC):
    """Base model for parametric single-drug dose response curves."""

    def __init__(self):
        """Ctor."""
        self.bounds = None
        self.fit_function = None
        self.jacobian_function = None

        self.converged = False
        self._fit = False

        self.sum_of_squares_residuals: Optional[float]
        self.r_squared: Optional[float]
        self.aic: Optional[float]
        self.bic: Optional[float]
        self.bootstrap_parameters = None

    @abstractmethod
    def _set_parameters(self, popt):
        """Internal method to set model parameters"""

    @abstractmethod
    def get_parameters(self) -> list:
        """Returns model parameters"""

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

    def fit(self, d, E, use_jacobian=True, bootstrap_iterations=0, **kwargs):
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
        self._fit = True
        d = np.asarray(d)
        E = np.asarray(E)

        # Are initial parameter guesses provided?
        if "p0" in kwargs:
            p0 = list(kwargs["p0"])
        else:
            p0 = None

        # Sanitize initial guesses
        p0 = self._get_initial_guess(d, E, p0=p0)
        kwargs["p0"] = p0

        with np.errstate(divide="ignore", invalid="ignore"):
            popt = self._internal_fit(d, E, use_jacobian, **kwargs)

        if popt is None:  # curve_fit() failed to fit parameters
            self.converged = False
            self._set_parameters(self._transform_params_from_fit(p0))
        else:  # curve_fit() succeeded
            self.converged = True
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
        if not self.converged:
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

    @property
    def is_specified(self):
        """True if all parameters are set"""
        parameters = self.get_parameters()
        if parameters is None:
            return False

        return None not in parameters and True not in np.isnan(np.asarray(parameters))

    def _internal_fit(self, d, E, use_jacobian: bool, **kwargs):
        """Fit the model to data (d, E)"""
        jac = self.jacobian_function if use_jacobian and self.jacobian_function else None

        popt = curve_fit(
            self.fit_function,
            d,
            E,
            bounds=self.bounds,
            jac=jac,
            **kwargs,
        )[0]

        if True in np.isnan(popt):
            return None
        return self._transform_params_from_fit(popt)

    def _get_initial_guess(self, d, E, p0=None):
        """Internal method to format and/or guess p0"""
        return p0

    def _transform_params_from_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform those to h and C
        """
        return params

    def _transform_params_to_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform from h and C
        """
        return params

    def _score(self, d, E):
        """Calculate goodness of fit and model quality scores

        This calculations
        - `sum_of_squares_residuals`
        - `r_squared`
        - `aic` (Akaike Information Criterion)
        - `bic` (Bayesian Information Criterion)

        Called automatically during model.fit(d1, d2, E)

        Parameters
        ----------
        d : array_like
            Doses

        E : array_like
            Measured dose-response at doses d
        """
        if self.is_specified:
            n_parameters = len(self.get_parameters())

            self.sum_of_squares_residuals = utils.residual_ss_1d(d, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, **kwargs):
        """Identify confidence intervals for parameters using bootstrap resampling.

        Residuals are randomly sampled from a normal distribution with :math:`\sigma = \sqrt{\frac{RSS}{n - N}}`
        where :math:`RSS` is the residual sum of square, :math:`n` is the number of data points, and :math:`N` is the
        number of parameters.
        """
        if not self.is_specified:
            raise ModelNotParameterizedError()
        if not self.converged:
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
                popt1 = self._internal_fit(d, E_iteration, use_jacobian=use_jacobian, **kwargs)

            if popt1 is not None:
                bootstrap_parameters.append(popt1)

        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

    def __repr__(self):
        name = type(self).__name__
        x: dict = {}  # TODO should map parameter name to value
        params = [f"{k}={v}" for k, v in x.items()]
        return f"{name}({', '.join(params)})"

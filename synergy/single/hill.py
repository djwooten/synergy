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

import numpy as np
from scipy.stats import linregress

from synergy.utils import base as utils
from synergy.exceptions import ModelNotParameterizedError
from synergy.single.parametric_base import ParameterizedModel1D


class Hill(ParameterizedModel1D):
    """The four-parameter Hill equation

                            d^h
    E = E0 + (Emax-E0) * ---------
                         C^h + d^h

    The Hill equation is a standard model for single-drug dose-response curves.
    This is the base model for Hill_2P and Hill_CI.

    """

    def __init__(
        self,
        E0=None,
        Emax=None,
        h=None,
        C=None,
        E0_bounds=None,
        Emax_bounds=None,
        h_bounds=None,
        C_bounds=None,
    ):
        """
        Parameters
        ----------
        E0 : float, optional
            Effect at 0 dose. Set this if you are creating a synthetic Hill
            model, rather than fitting from data

        Emax : float, optional
            Effect at 0 dose. Set this if you are creating a synthetic Hill
            model, rather than fitting from data

        h : float, optional
            The Hill-slope. Set this if you are creating a synthetic Hill
            model, rather than fitting from data

        C : float, optional
            EC50, the dose for which E = (E0+Emax)/2. Set this if you are
            creating a synthetic Hill model, rather than fitting from data

        X_bounds: tuple
            Bounds to use for Hill equation parameters during fitting. Valid options are E0_bounds, Emax_bounds, h_bounds, C_bounds.
        """

        super().__init__()

        if h is not None and h <= 0:
            raise ValueError(f"h must be > 0 ({h})")
        if C is not None and C <= 0:
            raise ValueError(f"C must be > 0 ({C})")
        self.E0 = E0
        self.Emax = Emax
        self.h = h
        self.C = C

        self.E0_bounds = E0_bounds if E0_bounds else (-np.inf, np.inf)
        self.Emax_bounds = Emax_bounds if Emax_bounds else (-np.inf, np.inf)
        self.h_bounds = h_bounds if h_bounds else (0, np.inf)
        self.C_bounds = C_bounds if C_bounds else (0, np.inf)

        # Transform h and C bounds to log scale
        with np.errstate(divide="ignore"):
            self.logh_bounds = (np.log(self.h_bounds[0]), np.log(self.h_bounds[1]))
            self.logC_bounds = (np.log(self.C_bounds[0]), np.log(self.C_bounds[1]))

        self.fit_function = self._model_to_fit
        self.jacobian_function = self._model_jacobian_for_fit

        self.bounds = tuple(zip(self.E0_bounds, self.Emax_bounds, self.logh_bounds, self.logC_bounds))

    def E(self, d):
        """Evaluate this model at dose d. If the model is not parameterized, returns 0.

        Parameters
        ----------
        d : array_like
            Doses to calculate effect at

        Returns
        ----------
        effect : array_like
            Evaluate's the model at dose in d
        """
        if not self.is_parameterized:
            return super().E(d)

        return self._model(d, self.E0, self.Emax, self.h, self.C)

    def E_inv(self, E):
        """Inverse of the Hill equation

        Parameters
        ----------
        E : array_like
            Effects to get the doses for

        Returns
        ----------
        doses : array_like
            Doses which achieve effects E using this model. Effects that are
            outside the range [E0, Emax] will return np.nan for the dose
        """
        if not self.is_parameterized:
            return super().E_inv(E)

        return self._model_inv(E, self.E0, self.Emax, self.h, self.C)

    def get_parameters(self):
        """Gets the model's parmaters.

        Returns
        ----------
        parameters : tuple
            (E0, Emax, h, C)
        """
        return (self.E0, self.Emax, self.h, self.C)

    def _set_parameters(self, popt):
        E0, Emax, h, C = popt

        self.E0 = E0
        self.Emax = Emax
        self.h = h
        self.C = C

    def _model(self, d, E0, Emax, h, C):
        dh = np.float_power(d, h)
        return E0 + (Emax - E0) * (dh / (C**h + dh))

    def _model_to_fit(self, d, E0, Emax, logh, logC):
        return self._model(d, E0, Emax, np.exp(logh), np.exp(logC))

    def _model_inv(self, E, E0, Emax, h, C):
        E_ratio = (E - E0) / (Emax - E)
        d = np.float_power(E_ratio, 1.0 / h) * C

        # For any E's outside of the range E0 to Emax, E_inv should be nan
        if hasattr(E, "__iter__"):
            d[E_ratio < 0] = np.nan
            return d

        # Dose cannot be negative
        elif d < 0:
            return np.nan

        return d

    def _model_jacobian_for_fit(self, d, E0, Emax, logh, logC):
        """
        Returns
        ----------
        jacobian : array_like
            Derivatives of the Hill equation with respect to E0, Emax, logh,
            and logC
        """
        h = np.exp(logh)
        d_pow_h = d**h
        C_pow_h = np.exp(logC) ** h
        logd = np.log(d)

        jE0 = 1 - d_pow_h / (C_pow_h + d_pow_h)
        jEmax = 1 - jE0

        jC = (E0 - Emax) * d_pow_h * h * C_pow_h / ((C_pow_h + d_pow_h) * (C_pow_h + d_pow_h))

        jh = (
            (Emax - E0)
            * d_pow_h
            * h
            * ((C_pow_h + d_pow_h) * logd - (logC * C_pow_h + logd * d_pow_h))
            / ((C_pow_h + d_pow_h) * (C_pow_h + d_pow_h))
        )

        jac = np.hstack((jE0.reshape(-1, 1), jEmax.reshape(-1, 1), jh.reshape(-1, 1), jC.reshape(-1, 1)))
        jac[np.isnan(jac)] = 0
        return jac

    def _get_initial_guess(self, d, E, p0=None):
        if p0 is None:
            p0 = [max(E), min(E), 1, np.median(d)]

        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        return params[0], params[1], np.exp(params[2]), np.exp(params[3])

    def _transform_params_to_fit(self, params):
        return params[0], params[1], np.log(params[2]), np.log(params[3])

    def __repr__(self):
        if not self.is_parameterized:
            return "Hill()"

        return "Hill(E0=%0.2f, Emax=%0.2f, h=%0.2f, C=%0.2e)" % (self.E0, self.Emax, self.h, self.C)


class Hill_2P(Hill):
    """The two-parameter Hill equation

                            d^h
    E = E0 + (Emax-E0) * ---------
                         C^h + d^h

    Mathematically equivalent to the four-parameter Hill equation, but E0 and Emax are held constant (not fit to data).

    """

    def __init__(self, E0=1, Emax=0, **kwargs):
        super().__init__(E0=E0, Emax=Emax, **kwargs)

        self.jacobian_function = self._model_jacobian_for_fit
        self.bounds = tuple(zip(self.logh_bounds, self.logC_bounds))

    def _model_to_fit(self, d, logh, logC):
        return self._model(d, self.E0, self.Emax, np.exp(logh), np.exp(logC))

    def _model_jacobian_for_fit(self, d, logh, logC):
        h = np.exp(logh)
        d_pow_h = d**h
        C_pow_h = np.exp(logC) ** h
        squared_sum = np.float_power(C_pow_h + d_pow_h, 2.0)

        logd = np.log(d)

        E0 = self.E0
        Emax = self.Emax

        jC = (E0 - Emax) * (d_pow_h * h * C_pow_h / squared_sum)

        jh = (Emax - E0) * d_pow_h * h * ((C_pow_h + d_pow_h) * logd - (logC * C_pow_h + logd * d_pow_h)) / squared_sum

        jac = np.hstack((jh.reshape(-1, 1), jC.reshape(-1, 1)))
        jac[np.isnan(jac)] = 0
        return jac

    def _get_initial_guess(self, d, E, p0=None):
        if p0 is None:
            p0 = [1, np.median(d)]

        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)

        return p0

    def get_parameters(self):
        """Gets the model's parameters

        Returns
        ----------
        parameters : tuple
            (h, C)
        """
        return (self.h, self.C)

    def _set_parameters(self, popt):
        h, C = popt

        self.h = h
        self.C = C

    def _transform_params_from_fit(self, params):
        return np.exp(params[0]), np.exp(params[1])

    def _transform_params_to_fit(self, params):
        return np.log(params[0]), np.log(params[1])

    def __repr__(self):
        if not self.is_parameterized:
            return "Hill_2P()"

        return "Hill_2P(E0=%0.2f, Emax=%0.2f, h=%0.2f, C=%0.2e)" % (
            self.E0,
            self.Emax,
            self.h,
            self.C,
        )


class Hill_CI(Hill_2P):
    """Mathematically equivalent two-parameter Hill equation with E0=1 and Emax=0. However, Hill_CI.fit() uses the log-linearization approach to dose-response fitting used by the Combination Index."""

    def __init__(self, h=None, C=None):
        super().__init__(h=h, C=C, E0=1.0, Emax=0.0)

    def _internal_fit(self, d, E, use_jacobian, **kwargs):
        """Override the parent function to use linregress() instead of curve_fit()"""
        mask = np.where((E < 1) & (E > 0) & (d > 0))
        E = E[mask]
        d = d[mask]
        fU = E
        fA = 1 - E

        median_effect_line = linregress(np.log(d), np.log(fA / fU))
        h = median_effect_line.slope
        C = np.exp(-median_effect_line.intercept / h)

        return (h, C)

    def plot_linear_fit(self, d, E, ax=None):
        if not self.is_parameterized:
            raise ModelNotParameterizedError()

        try:
            from matplotlib import pyplot as plt
        except:
            # TODO: Error
            # TODO: Move this whole function to utils.plot
            return
        mask = np.where((E < 1) & (E > 0) & (d > 0))
        E = E[mask]
        d = d[mask]
        fU = E
        fA = 1 - E

        ax_created = False
        if ax is None:
            ax_created = True
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)

        ax.scatter(np.log(d), np.log(fA / fU))
        ax.plot(np.log(d), np.log(d) * self.h - self.h * np.log(self.C))

        # Draw bootstraps with low opacity
        # for i in range(self.bootstrap_parameters.shape[0]):
        #     h, C = self.bootstrap_parameters[i, :]
        #     ax.plot(np.log(d), np.log(d) * h - h * np.log(C), c="k", alpha=0.1, lw=0.5)

        ax.set_ylabel("h * log(d) - h * log(C)")
        ax.set_xlabel("log(d)")
        ax.set_title("Combination Index linearization")
        if ax_created:
            plt.tight_layout()
            plt.show()

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, confidence_interval, **kwargs):
        """Bootstrap resampling is not yet implemented for CI"""

    def __repr__(self):
        if not self.is_parameterized:
            return "Hill_CI()"

        return "Hill_CI(h=%0.2f, C=%0.2e)" % (self.h, self.C)

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

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import linregress

from synergy.exceptions import ModelNotParameterizedError
from synergy.single.dose_response_model_1d import ParametricDoseResponseModel1D


class Hill(ParametricDoseResponseModel1D):
    """The four-parameter Hill equation

    E = E0 + (Emax - E0) * d^h / (C^h + d^h)

    The Hill equation is a standard model for single-drug dose-response curves.
    This is the base model for Hill_2P and Hill_CI.
    """

    def __init__(self, **kwargs):
        """Ctor."""
        # To minimize risk of overflow or floating-point precision issues, we linearly scale
        # doses passed into fit to be centered around 0 on a log scale.
        # This variable stores that scale and is used to reverse it when fitting C.
        self._dose_scale: float = 1.0

        self.fit_function = self._model_to_fit
        self.jacobian_function = self._model_jacobian_for_fit

        super().__init__(**kwargs)

    def E(self, d):
        if not self.is_specified:
            raise ModelNotParameterizedError("Model mustbe specified before calling E().")

        return self._model(d, self.E0, self.Emax, self.h, self.C)

    def E_inv(self, E):
        if not self.is_specified:
            raise ModelNotParameterizedError("Model mustbe specified before calling E().")

        return self._model_inv(E, self.E0, self.Emax, self.h, self.C)

    @property
    def _parameter_names(self) -> List[str]:
        return ["E0", "Emax", "h", "C"]

    @property
    def _default_fit_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {"h": (0.0, np.inf), "C": (0.0, np.inf)}

    def _set_dose_scale(self, d):
        """Find the scaling factor that will normalize the dose scale to be log-centered around 0.

        We want to find s such that d' = d / s has
        mean(log(d')) = 0

        We will then use d' when fitting the data.
        Importantly, this will mean we are fitting C' = C / s, not just C.
        Consequently we must update C_bounds to C'_bounds.
        """
        C_idx = self._parameter_names.index("C")
        # Revert old C'_bounds back to C_bounds
        if C_idx >= 0:
            self._bounds[0][C_idx] += np.log(self._dose_scale)
            self._bounds[1][C_idx] += np.log(self._dose_scale)

        # Calculate dose scale
        self._dose_scale = np.exp(np.median(np.log(d[d > 0])))

        # Update C_bounds to C'_bounds
        if C_idx >= 0:
            self._bounds[0][C_idx] -= np.log(self._dose_scale)
            self._bounds[1][C_idx] -= np.log(self._dose_scale)

    def fit(self, d, E, use_jacobian=True, bootstrap_iterations=0, **kwargs):
        self._set_dose_scale(d)
        super().fit(
            d / self._dose_scale, E, use_jacobian=use_jacobian, bootstrap_iterations=bootstrap_iterations, **kwargs
        )

    def _set_parameters(self, parameters):
        self.E0, self.Emax, self.h, self.C = parameters

    def _model(self, d, E0, Emax, h, C):
        """Hill equation."""
        dh = np.float_power(d, h)
        return E0 + (Emax - E0) * (dh / (C**h + dh))

    def _model_to_fit(self, d, E0, Emax, logh, logC):
        """Hill equation expecting log-transformed parameters h and C parameters, for fitting."""
        return self._model(d, E0, Emax, np.exp(logh), np.exp(logC))

    def _model_inv(self, E, E0, Emax, h, C):
        """Inverse Hill equation."""
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
        """Hill equation jacobian, expecting log-transformed h and C, for fitting.

        Return
        ------
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

    def _get_initial_guess(self, d, E, p0):
        """Default initial guess is E0=E(dmin), Emax=E(dmax), h=1, C=median(d)"""
        if p0 is None:
            p0 = [np.median(E[d == min(d)]), np.median(E[d == max(d)]), 1, np.median(d) * self._dose_scale]

        return super()._get_initial_guess(d, E, p0)

    def _transform_params_from_fit(self, params):
        """Exponentiate h and C to get the final parameters."""
        return params[0], params[1], np.exp(params[2]), np.exp(params[3]) * self._dose_scale

    def _transform_params_to_fit(self, params):
        """Log-transform h and C for fitting."""
        with np.errstate(divide="ignore"):
            return params[0], params[1], np.log(params[2]), np.log(params[3] / self._dose_scale)

    def __repr__(self):
        if not self.is_specified:
            return "Hill()"

        return "Hill(E0=%0.3g, Emax=%0.3g, h=%0.3g, C=%0.3g)" % (self.E0, self.Emax, self.h, self.C)


class Hill_2P(Hill):
    """The two-parameter Hill equation

    E = E0 + (Emax - E0) * d^h / (C^h + d^h)

    Mathematically equivalent to the four-parameter Hill equation, but E0 and Emax are held constant (not fit to data).
    """

    def __init__(self, E0=1.0, Emax=0.0, **kwargs):
        self.E0 = E0
        self.Emax = Emax
        super().__init__(**kwargs)

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

    @property
    def _parameter_names(self) -> List[str]:
        return ["h", "C"]

    def _get_initial_guess(self, d, E, p0):
        if p0 is None:
            p0 = [1, np.median(d)]

        return super()._get_initial_guess(d, E, p0)

    def _set_parameters(self, popt):
        h, C = popt

        self.h = h
        self.C = C

    def _transform_params_from_fit(self, params):
        return np.exp(params[0]), np.exp(params[1]) * self._dose_scale

    def _transform_params_to_fit(self, params):
        with np.errstate(divide="ignore"):
            return np.log(params[0]), np.log(params[1] / self._dose_scale)

    def __repr__(self):
        if not self.is_specified:
            return "Hill_2P()"

        return "Hill_2P(E0=%0.3g, Emax=%0.3g, h=%0.3g, C=%0.3g)" % (
            self.E0,
            self.Emax,
            self.h,
            self.C,
        )


class Hill_CI(Hill_2P):
    """Model used to calculate Combination Index synergy.

    Mathematically this equivalent two-parameter Hill equation with E0=1 and Emax=0. However, Hill_CI.fit() uses a
    log-linearization approach to dose-response fitting used by the Combination Index.
    """

    def __init__(self, **kwargs):
        kwargs["E0"] = 1.0
        kwargs["Emax"] = 0.0
        super().__init__(**kwargs)

    def _fit(self, d, E, use_jacobian, **kwargs):
        """Override the parent function to use linregress() instead of curve_fit()"""
        mask = np.where((E < 1) & (E > 0) & (d > 0))
        E = E[mask]
        d = d[mask]
        fU = E
        fA = 1 - E

        median_effect_line = linregress(np.log(d), np.log(fA / fU))
        h = median_effect_line.slope
        C = np.exp(-median_effect_line.intercept / h)
        C *= self._dose_scale
        return (h, C)

    def plot_linear_fit(self, d, E, ax=None):
        if not self.is_specified:
            raise ModelNotParameterizedError()

        try:
            from matplotlib import pyplot as plt
        except ImportError:
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

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, **kwargs):
        """Bootstrap resampling is not yet implemented for CI"""

    def __repr__(self):
        if not self.is_specified:
            return "Hill_CI()"

        return "Hill_CI(h=%0.3g, C=%0.3g)" % (self.h, self.C)

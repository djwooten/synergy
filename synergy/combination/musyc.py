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

from .jacobians.musyc_jacobian import jacobian
from .parametric_base import ParametricModel

from synergy.utils import base as utils
from synergy.single import Hill
from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError


class MuSyC(ParametricModel):
    """The MuSyC parametric synergy model for combinations of two drugs.

    Multidimensional Synergy of Combinations (MuSyC) is a drug synergy framework based on the law of mass action
    (`doi: 10.1016/j.cels.2019.01.003 <https://doi.org/10.1016/j.cels.2019.01.003>`_, `doi: 10.1101/683433 <https://doi.org/10.1101/683433>`_).

    In MuSyC, synergy is parametrically defined as shifts in potency (alpha), efficacy (beta), or cooperativity (gamma).

    .. csv-table:: Interpretation of synergy parameters
       :header: "Parameter", "Values", "Synergy/Antagonism", "Interpretation"

       "``alpha12``", "[0, 1)", "Antagonistic Potency",       "Drug 1 decreases the effective dose (potency) of drug 2"
       ,              "> 1",    "Synergistic Potency",        "Drug 1 increases the effective dose (potency) of drug 2"
       "``alpha21``", "[0, 1)", "Antagonistic Potency",       "Drug 2 decreases the effective dose (potency) of drug 1"
       ,              "> 1",    "Synergistic Potency",        "Drug 2 increases the effective dose (potency) of drug 1"
       "``beta``",    "< 0",    "Antagonistic Efficacy",      "The combination is weaker than the stronger of drugs 1 and 2"
       ,              "> 0",    "Synergistic Efficacy",       "The combination is stronger than the stronger of drugs 1 and 2"
       "``gamma12``", "[0, 1)", "Antagonistic Cooperativity", "Drug 1 decreases the effective dose (potency) of drug 2"
       ,              "> 1",    "Synergistic Cooperativity",  "Drug 1 increases the effective dose (potency) of drug 2"
       "``gamma21``", "[0, 1)", "Antagonistic Cooperativity", "Drug 2 decreases the effective dose (potency) of drug 1"
       ,              "> 1",    "Synergistic Cooperativity",  "Drug 2 increases the effective dose (potency) of drug 1"
    """

    def __init__(
        self,
        drug1_model=None,
        drug2_model=None,
        alpha12=None,
        alpha21=None,
        E3=None,
        gamma12=None,
        gamma21=None,
        r1r=1.0,
        r2r=1.0,
        fit_gamma=True,
    ):
        """Ctor.

        :param float alpha12: Synergistic potency of drug 1 on drug 2 ([0, 1) = antagonism, (1, inf) = synergism)
        :param float alpha21: Synergistic potency of drug 2 on drug 1 ([0, 1) = antagonism, (1, inf) = synergism).
        :param float beta: Synergistic efficacy ((-inf,0) = antagonism, (0,inf) = synergism)
        :param float gamma12: Synergistic cooperativity of drug 1 on drug 2 ([0,1) = antagonism, (1,inf) = synergism)
        :param float gamma21: Synergistic cooperativity of drug 2 on drug 1 ([0,1) = antagonism, (1,inf) = synergism)
        """
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model)

        self.fit_gamma = fit_gamma

        self.r1r = r1r
        self.r2r = r2r

        self.alpha12 = alpha12
        self.alpha21 = alpha21
        self.E3 = E3
        self.gamma12 = gamma12
        self.gamma21 = gamma21

        if fit_gamma:
            self.fit_function = self._model_to_fit_with_gamma
            self.jacobian_function = self._jacobian_with_gamma

        else:
            self.fit_function = self._model_to_fit_no_gamma
            self.jacobian_function = self._jacobian_no_gamma

    @property
    def beta(self):
        """-"""
        E0 = (self.drug1_model.E0 + self.drug2_model.E0) / 2.0
        return MuSyC._get_beta(E0, self.drug1_model.Emax, self.drug2_model.Emax, self.E3)

    def _model_to_fit_with_gamma(
        self, d, E0, E1, E2, E3, h1, h2, C1, C2, logalpha12, logalpha21, loggamma12, loggamma21
    ):
        return self._model(
            d[0],
            d[1],
            E0,
            E1,
            E2,
            E3,
            h1,
            h2,
            C1,
            C2,
            self.r1r,
            self.r2r,
            np.exp(logalpha12),
            np.exp(logalpha21),
            np.exp(loggamma12),
            np.exp(loggamma21),
        )

    def _model_to_fit_no_gamma(self, d, E0, E1, E2, E3, h1, h2, C1, C2, logalpha12, logalpha21):
        return self._model(
            d[0], d[1], E0, E1, E2, E3, h1, h2, C1, C2, self.r1r, self.r2r, np.exp(logalpha12), np.exp(logalpha21), 1, 1
        )

    def _jacobian_with_gamma(
        self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21
    ):
        """Calculate the jacobian inlcuding gamma.

        Derivatives in the jacobian are already defined with respect to (e.g.) log(h) or log(alpha), rather than the
        linear values, so np.exp() is not required (or desired) here.
        """
        return jacobian(
            d[0],
            d[1],
            E0,
            E1,
            E2,
            E3,
            logh1,
            logh2,
            logC1,
            logC2,
            self.r1r,
            self.r2r,
            logalpha12,
            logalpha21,
            loggamma12,
            loggamma21,
        )

    def _jacobian_no_gamma(self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21):
        """Calculate the jacobian assuming gamma==1.

        Derivatives in the jacobian are already defined with respect to (e.g.) log(h) or log(alpha), rather than the
        linear values, so np.exp() is not required (or desired) here.

        The [:, :, -2] gets rid of derivatives WRT gamma
        TODO: Speed things up by defining a no_gamma jacobian that doesn't even calculate them at all
        """
        return jacobian(
            d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1r, self.r2r, logalpha12, logalpha21, 0, 0
        )[:, :, -2]

    def _get_initial_guess(self, d1, d2, E, p0=None):

        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:

            # Fit the single drug models if they were not pre-fit by the user
            if not self.drug1_model.is_specified:
                mask = np.where(d2 == min(d2))
                self.drug1_model.fit(d1[mask], E[mask])
            if not self.drug2_model.is_specified:
                mask = np.where(d1 == min(d1))
                self.drug2_model.fit(d2[mask], E[mask])

            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = self.drug1_model.get_parameters()
            E0_2, E2, h2, C2 = self.drug2_model.get_parameters()

            # Get initial guess of E3 at E(d1_max, d2_max), if that point exists
            # It may not exist if the input data are not sampled on a regular grid
            E3 = E[(d1 == max(d1)) & (d2 == max(d2))]
            if len(E3) > 0:
                E3 = np.mean(E3)

            # TODO: E orientation
            # Otherwise guess E3 is the minimum E observed
            else:
                E3 = np.min(E)

            p0 = [(E0_1 + E0_2) / 2.0, E1, E2, E3, h1, h2, C1, C2, 1, 1, 1, 1]

            if not self.fit_gamma:
                p0 = p0[:-2]

        p0 = list(self._transform_params_to_fit(p0))
        bounds = ()  # TODO: Redo bounds later
        utils.sanitize_initial_guess(p0, bounds)
        return p0

    def _transform_params_from_fit(self, params):
        """Transforms logscaled parameters to linear scale"""
        if not self.fit_gamma:
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21 = params
        else:
            (
                E0,
                E1,
                E2,
                E3,
                logh1,
                logh2,
                logC1,
                logC2,
                logalpha12,
                logalpha21,
                loggamma12,
                loggamma21,
            ) = params
            gamma12 = np.exp(loggamma12)
            gamma21 = np.exp(loggamma21)

        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        alpha12 = np.exp(logalpha12)
        alpha21 = np.exp(logalpha21)

        if not self.fit_gamma:
            return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21

        return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21

    def _transform_params_to_fit(self, params):
        """Transform appropriate linear params to log-scale for fitting"""
        if not self.fit_gamma:
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21 = params
        else:
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21 = params
            loggamma12 = np.log(gamma12)
            loggamma21 = np.log(gamma21)

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)
        logalpha12 = np.log(alpha12)
        logalpha21 = np.log(alpha21)

        if not self.fit_gamma:
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21

        return (
            E0,
            E1,
            E2,
            E3,
            logh1,
            logh2,
            logC1,
            logC2,
            logalpha12,
            logalpha21,
            loggamma12,
            loggamma21,
        )

    def E(self, d1, d2):
        if not self.is_specified:
            raise ModelNotParameterizedError()

        if not self.fit_gamma:
            return self._model(
                d1,
                d2,
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.r1r,
                self.r2r,
                self.alpha12,
                self.alpha21,
                1,
                1,
            )

        else:
            return self._model(
                d1,
                d2,
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.r1r,
                self.r2r,
                self.alpha12,
                self.alpha21,
                self.gamma12,
                self.gamma21,
            )

    def _get_parameters(self):
        if not self.fit_gamma:
            return (
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.alpha12,
                self.alpha21,
            )
        else:
            return (
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.alpha12,
                self.alpha21,
                self.gamma12,
                self.gamma21,
            )

    def _set_parameters(self, popt):
        if not self.fit_gamma:
            (
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.alpha12,
                self.alpha21,
            ) = popt
        else:
            (
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.alpha12,
                self.alpha21,
                self.gamma12,
                self.gamma21,
            ) = popt

    def _reference_E(self, d1, d2):
        if not self.is_specified:
            raise ModelNotParameterizedError()

        return self._model(
            d1,
            d2,
            self.E0,
            self.E1,
            self.E2,
            min(self.E1, self.E2),
            self.h1,
            self.h2,
            self.C1,
            self.C2,
            self.r1r,
            self.r2r,
            1,
            1,
            1,
            1,
        )

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, r1r, r2r, alpha12, alpha21, gamma12, gamma21):
        # Precompute some terms that are used repeatedly
        d1_pow_h1 = np.power(d1, h1)
        d2_pow_h2 = np.power(d2, h2)
        C1_pow_h1 = np.power(C1, h1)
        C2_pow_h2 = np.power(C2, h2)

        r1 = r1r / C1_pow_h1
        r2 = r2r / C2_pow_h2

        alpha21_d1_pow_gamma21_h1 = np.power(alpha21 * d1, gamma21 * h1)
        alpha12_d2_pow_gamma12_h2 = np.power(alpha12 * d2, gamma12 * h2)
        r1_C1h1_pow_gamma21 = np.power((r1 * C1_pow_h1), gamma21)
        r2_C2h2_pow_gamma12 = np.power((r2 * C2_pow_h2), gamma12)
        r1_pow_gamma21_plus_1 = np.power(r1, (gamma21 + 1))
        r2_pow_gamma12_plus_1 = np.power(r2, (gamma12 + 1))
        r1_pow_gamma21 = np.power(r1, gamma21)
        r2_pow_gamma12 = np.power(r2, gamma12)

        # Unaffected population
        U = (
            r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1 * C2_pow_h2
            + r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1 * C2_pow_h2
            + r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21 * C2_pow_h2
        ) / (
            d1_pow_h1 * r1 * r2 * r1_C1h1_pow_gamma21 * C2_pow_h2
            + d1_pow_h1 * r1 * r2 * r2_C2h2_pow_gamma12 * C2_pow_h2
            + d1_pow_h1 * r1 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * C2_pow_h2
            + d1_pow_h1 * r1 * r2_pow_gamma12 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * r2_pow_gamma12 * alpha21_d1_pow_gamma21_h1 * alpha12_d2_pow_gamma12_h2
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1
            + d2_pow_h2 * r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + d2_pow_h2 * r1_pow_gamma21_plus_1 * r2 * alpha21_d1_pow_gamma21_h1 * C1_pow_h1
            + d2_pow_h2 * r1_pow_gamma21 * r2 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1_pow_gamma21 * r2_pow_gamma12_plus_1 * alpha21_d1_pow_gamma21_h1 * alpha12_d2_pow_gamma12_h2
            + d2_pow_h2 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1 * C2_pow_h2
            + r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1 * C2_pow_h2
            + r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21 * C2_pow_h2
        )

        # Affected by drug 1 only
        A1 = (
            d1_pow_h1 * r1 * r2 * r1_C1h1_pow_gamma21 * C2_pow_h2
            + d1_pow_h1 * r1 * r2 * r2_C2h2_pow_gamma12 * C2_pow_h2
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1_pow_gamma21 * r2 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
        ) / (
            d1_pow_h1 * r1 * r2 * r1_C1h1_pow_gamma21 * C2_pow_h2
            + d1_pow_h1 * r1 * r2 * r2_C2h2_pow_gamma12 * C2_pow_h2
            + d1_pow_h1 * r1 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * C2_pow_h2
            + d1_pow_h1 * r1 * r2_pow_gamma12 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * r2_pow_gamma12 * alpha21_d1_pow_gamma21_h1 * alpha12_d2_pow_gamma12_h2
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1
            + d2_pow_h2 * r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + d2_pow_h2 * r1_pow_gamma21_plus_1 * r2 * alpha21_d1_pow_gamma21_h1 * C1_pow_h1
            + d2_pow_h2 * r1_pow_gamma21 * r2 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1_pow_gamma21 * r2_pow_gamma12_plus_1 * alpha21_d1_pow_gamma21_h1 * alpha12_d2_pow_gamma12_h2
            + d2_pow_h2 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1 * C2_pow_h2
            + r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1 * C2_pow_h2
            + r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21 * C2_pow_h2
        )

        # Affected by drug 2 only
        A2 = (
            d1_pow_h1 * r1 * r2_pow_gamma12 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + d2_pow_h2 * r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1
            + d2_pow_h2 * r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + d2_pow_h2 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
        ) / (
            d1_pow_h1 * r1 * r2 * r1_C1h1_pow_gamma21 * C2_pow_h2
            + d1_pow_h1 * r1 * r2 * r2_C2h2_pow_gamma12 * C2_pow_h2
            + d1_pow_h1 * r1 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * C2_pow_h2
            + d1_pow_h1 * r1 * r2_pow_gamma12 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * r2_pow_gamma12 * alpha21_d1_pow_gamma21_h1 * alpha12_d2_pow_gamma12_h2
            + d1_pow_h1 * r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1
            + d2_pow_h2 * r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + d2_pow_h2 * r1_pow_gamma21_plus_1 * r2 * alpha21_d1_pow_gamma21_h1 * C1_pow_h1
            + d2_pow_h2 * r1_pow_gamma21 * r2 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12
            + d2_pow_h2 * r1_pow_gamma21 * r2_pow_gamma12_plus_1 * alpha21_d1_pow_gamma21_h1 * alpha12_d2_pow_gamma12_h2
            + d2_pow_h2 * r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21
            + r1 * r2 * r1_C1h1_pow_gamma21 * C1_pow_h1 * C2_pow_h2
            + r1 * r2 * r2_C2h2_pow_gamma12 * C1_pow_h1 * C2_pow_h2
            + r1_pow_gamma21_plus_1 * alpha21_d1_pow_gamma21_h1 * r2_C2h2_pow_gamma12 * C1_pow_h1
            + r2_pow_gamma12_plus_1 * alpha12_d2_pow_gamma12_h2 * r1_C1h1_pow_gamma21 * C2_pow_h2
        )

        # Affected by both drugs
        A3 = 1 - (U + A1 + A2)

        return U * E0 + A1 * E1 + A2 * E2 + A3 * E3

    @staticmethod
    def _get_beta(E0, E1, E2, E3):
        """Calculate synergistic efficacy."""
        strongest_E = np.amin(np.asarray([E1, E2]), axis=0)
        beta = (strongest_E - E3) / (E0 - strongest_E)
        return beta

    def get_parameters(self, confidence_interval=95):
        if not self.is_specified:
            raise ModelNotParameterizedError()

        if self.converged and self.bootstrap_parameters is not None:
            parameter_ranges = self.get_parameter_range(confidence_interval=confidence_interval)
        else:
            parameter_ranges = None

        params = {}
        params["E0"] = [
            self.E0,
        ]
        params["E1"] = [
            self.E1,
        ]
        params["E2"] = [
            self.E2,
        ]
        params["E3"] = [
            self.E3,
        ]
        params["h1"] = [
            self.h1,
        ]
        params["h2"] = [
            self.h2,
        ]
        params["C1"] = [
            self.C1,
        ]
        params["C2"] = [
            self.C2,
        ]
        params["beta"] = [
            self.beta,
        ]
        params["alpha12"] = [
            self.alpha12,
        ]
        params["alpha21"] = [
            self.alpha21,
        ]
        if self.fit_gamma:
            params["gamma12"] = [
                self.gamma12,
            ]
            params["gamma21"] = [
                self.gamma21,
            ]

        if parameter_ranges is not None:
            params["E0"].append(parameter_ranges[:, 0])
            params["E1"].append(parameter_ranges[:, 1])
            params["E2"].append(parameter_ranges[:, 2])
            params["E3"].append(parameter_ranges[:, 3])
            params["h1"].append(parameter_ranges[:, 4])
            params["h2"].append(parameter_ranges[:, 5])
            params["C1"].append(parameter_ranges[:, 6])
            params["C2"].append(parameter_ranges[:, 7])
            params["alpha12"].append(parameter_ranges[:, 8])
            params["alpha21"].append(parameter_ranges[:, 9])
            if self.fit_gamma:
                params["gamma12"].append(parameter_ranges[:, 10])
                params["gamma21"].append(parameter_ranges[:, 11])

            bsE0 = self.bootstrap_parameters[:, 0]  # type: ignore
            bsE1 = self.bootstrap_parameters[:, 1]  # type: ignore
            bsE2 = self.bootstrap_parameters[:, 2]  # type: ignore
            bsE3 = self.bootstrap_parameters[:, 3]  # type: ignore
            beta_bootstrap = MuSyC._get_beta(bsE0, bsE1, bsE2, bsE3)

            beta_bootstrap = np.percentile(
                beta_bootstrap, [(100 - confidence_interval) / 2, 50 + confidence_interval / 2]
            )
            params["beta"].append(beta_bootstrap)
        return params

    def summarize(self, confidence_interval: float = 95, tol: float = 0.01):
        """-"""
        pars = self.get_parameters(confidence_interval=confidence_interval)
        if pars is None:
            return None

        ret = []
        keys = pars.keys()
        # beta
        for key in keys:
            if "beta" in key:
                l = pars[key]
                if len(l) == 1:
                    if l[0] < -tol:
                        ret.append("%s\t%0.2f\t(<0) antagonistic" % (key, l[0]))
                    elif l[0] > tol:
                        ret.append("%s\t%0.2f\t(>0) synergistic" % (key, l[0]))
                else:
                    v = l[0]
                    lb, ub = l[1]
                    if v < -tol and lb < -tol and ub < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<0) antagonistic" % (key, v, lb, ub))
                    elif v > tol and lb > tol and ub > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>0) synergistic" % (key, v, lb, ub))
        # alpha
        for key in keys:
            if "alpha" in key:
                l = pars[key]
                if len(l) == 1:
                    if np.log10(l[0]) < -tol:
                        ret.append("%s\t%0.2f\t(<1) antagonistic" % (key, l[0]))
                    elif np.log10(l[0]) > tol:
                        ret.append("%s\t%0.2f\t(>1) synergistic" % (key, l[0]))
                else:
                    v = l[0]
                    lb, ub = l[1]
                    if np.log10(v) < -tol and np.log10(lb) < -tol and np.log10(ub) < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<1) antagonistic" % (key, v, lb, ub))
                    elif np.log10(v) > tol and np.log10(lb) > tol and np.log10(ub) > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>1) synergistic" % (key, v, lb, ub))

        # gamma
        for key in keys:
            if "gamma" in key:
                l = pars[key]
                if len(l) == 1:
                    if np.log10(l[0]) < -tol:
                        ret.append("%s\t%0.2f\t(<1) antagonistic" % (key, l[0]))
                    elif np.log10(l[0]) > tol:
                        ret.append("%s\t%0.2f\t(>1) synergistic" % (key, l[0]))
                else:
                    v = l[0]
                    lb, ub = l[1]
                    if np.log10(v) < -tol and np.log10(lb) < -tol and np.log10(ub) < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<1) antagonistic" % (key, v, lb, ub))
                    elif np.log10(v) > tol and np.log10(lb) > tol and np.log10(ub) > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>1) synergistic" % (key, v, lb, ub))
        if len(ret) > 0:
            return "\n".join(ret)
        else:
            return "No synergy or antagonism detected with %d percent confidence interval" % (int(confidence_interval))

    def __repr__(self):
        if not self.is_specified:
            return "MuSyC()"

        # beta = (min(self.E1,self.E2)-self.E3) / (self.E0 - min(self.E1,self.E2))
        beta = MuSyC._get_beta(self.E0, self.E1, self.E2, self.E3)

        if not self.fit_gamma:
            return (
                "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f)"
                % (
                    self.E0,
                    self.E1,
                    self.E2,
                    self.E3,
                    self.h1,
                    self.h2,
                    self.C1,
                    self.C2,
                    self.alpha12,
                    self.alpha21,
                    beta,
                )
            )
        return (
            "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f, gamma12=%0.2f, gamma21=%0.2f)"
            % (
                self.E0,
                self.E1,
                self.E2,
                self.E3,
                self.h1,
                self.h2,
                self.C1,
                self.C2,
                self.alpha12,
                self.alpha21,
                beta,
                self.gamma12,
                self.gamma21,
            )
        )

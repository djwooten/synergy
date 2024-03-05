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

from synergy.combination.jacobians.musyc_jacobian import jacobian
from synergy.combination.synergy_model_2d import ParametricSynergyModel2D
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single import Hill
from synergy.utils.base import format_table
from synergy.exceptions import ModelNotParameterizedError


class MuSyC(ParametricSynergyModel2D):
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

    def __init__(self, drug1_model=None, drug2_model=None, r1r=1.0, r2r=1.0, fit_gamma=True, **kwargs):
        """Ctor.

        :param float alpha12: Synergistic potency of drug 1 on drug 2 ([0, 1) = antagonism, (1, inf) = synergism)
        :param float alpha21: Synergistic potency of drug 2 on drug 1 ([0, 1) = antagonism, (1, inf) = synergism).
        :param float beta: Synergistic efficacy ((-inf,0) = antagonism, (0,inf) = synergism)
        :param float gamma12: Synergistic cooperativity of drug 1 on drug 2 ([0,1) = antagonism, (1,inf) = synergism)
        :param float gamma21: Synergistic cooperativity of drug 2 on drug 1 ([0,1) = antagonism, (1,inf) = synergism)
        """
        self.fit_gamma = fit_gamma
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)

        self.r1r = r1r
        self.r2r = r2r

        if fit_gamma:
            self.fit_function = self._model_to_fit_with_gamma
            self.jacobian_function = self._jacobian_with_gamma

        else:
            self.fit_function = self._model_to_fit_no_gamma
            self.jacobian_function = self._jacobian_no_gamma
            self.gamma12 = 1.0
            self.gamma21 = 1.0

        # beta is not a parameter used in the curve_fit, rather it is based on E0, E1, E2, and E3. Thus it is not fit
        # as a standard part of bootstrapping. So to get confidence intervals, we must handle it separately.
        self.bootstrap_beta = None

    @property
    def _parameter_names(self) -> list[str]:
        """-"""
        if self.fit_gamma:
            return ["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "alpha12", "alpha21", "gamma12", "gamma21"]
        return ["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "alpha12", "alpha21"]

    @property
    def _default_fit_bounds(self) -> dict[str, tuple[float, float]]:
        """-"""
        return {
            "h1": (0, np.inf),
            "h2": (0, np.inf),
            "C1": (0, np.inf),
            "C2": (0, np.inf),
            "alpha12": (0, np.inf),
            "alpha21": (0, np.inf),
            "gamma12": (0, np.inf),
            "gamma21": (0, np.inf),
        }

    def E_reference(self, d1, d2):
        """-"""
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
            1.0,
            1.0,
            1.0,
            1.0,
        )

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return Hill

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return Hill

    @property
    def _default_drug1_kwargs(self) -> dict:
        """-"""
        lb, ub = self._bounds
        param_names = self._parameter_names
        E0_idx = param_names.index("E0")
        Emax_idx = param_names.index("E1")
        h_idx = param_names.index("h1")
        C_idx = param_names.index("C1")
        return {
            "E0_bounds": (lb[E0_idx], ub[E0_idx]),
            "Emax_bounds": (lb[Emax_idx], ub[Emax_idx]),
            "h_bounds": (np.exp(lb[h_idx]), np.exp(ub[h_idx])),
            "C_bounds": (np.exp(lb[C_idx]), np.exp(ub[C_idx])),
        }

    @property
    def _default_drug2_kwargs(self) -> dict:
        """-"""
        lb, ub = self._bounds
        param_names = self._parameter_names
        E0_idx = param_names.index("E0")
        Emax_idx = param_names.index("E2")
        h_idx = param_names.index("h2")
        C_idx = param_names.index("C2")
        return {
            "E0_bounds": (lb[E0_idx], ub[E0_idx]),
            "Emax_bounds": (lb[Emax_idx], ub[Emax_idx]),
            "h_bounds": (np.exp(lb[h_idx]), np.exp(ub[h_idx])),
            "C_bounds": (np.exp(lb[C_idx]), np.exp(ub[C_idx])),
        }

    @property
    def beta(self):
        """-"""
        return MuSyC._get_beta(self.E0, self.E1, self.E2, self.E3)

    def _model_to_fit_with_gamma(
        self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21
    ):
        return self._model(
            d[0],
            d[1],
            E0,
            E1,
            E2,
            E3,
            np.exp(logh1),
            np.exp(logh2),
            np.exp(logC1),
            np.exp(logC2),
            self.r1r,
            self.r2r,
            np.exp(logalpha12),
            np.exp(logalpha21),
            np.exp(loggamma12),
            np.exp(loggamma21),
        )

    def _model_to_fit_no_gamma(self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21):
        return self._model(
            d[0],
            d[1],
            E0,
            E1,
            E2,
            E3,
            np.exp(logh1),
            np.exp(logh2),
            np.exp(logC1),
            np.exp(logC2),
            self.r1r,
            self.r2r,
            np.exp(logalpha12),
            np.exp(logalpha21),
            1,
            1,
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

        The [:, :-2] gets rid of derivatives WRT gamma
        TODO: Speed things up by defining a no_gamma jacobian that doesn't even calculate them at all
        """
        return jacobian(
            d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1r, self.r2r, logalpha12, logalpha21, 0, 0
        )[:, :-2]

    def _get_initial_guess(self, d1, d2, E, p0):
        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:

            drug1 = self.drug1_model
            drug2 = self.drug2_model

            if not (isinstance(drug1, Hill) and isinstance(drug2, Hill)):
                raise ValueError("Wrong single drug types")

            # Fit the single drug models if they were not pre-specified by the user
            if not drug1.is_specified:
                mask = np.where(d2 == min(d2))
                drug1.fit(d1[mask], E[mask])
            if not drug2.is_specified:
                mask = np.where(d1 == min(d1))
                drug2.fit(d2[mask], E[mask])

            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = drug1.E0, drug1.Emax, drug1.h, drug1.C
            E0_2, E2, h2, C2 = drug2.E0, drug2.Emax, drug2.h, drug2.C
            E0 = (E0_1 + E0_2) / 2.0

            # Get initial guess of E3 at E(d1_max, d2_max), if that point exists
            # It may not exist if the input data are not sampled on a regular grid
            E3 = E[(d1 == max(d1)) & (d2 == max(d2))]
            if len(E3) > 0:
                E3 = np.median(E3)

            # TODO: E orientation
            # Otherwise guess E3 is the minimum E observed
            else:
                E3 = np.min(E)

            p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 1, 1, 1, 1]

            if not self.fit_gamma:
                p0 = p0[:-2]

        return super()._get_initial_guess(d1, d2, E, p0)

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

        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        alpha12 = np.exp(logalpha12)
        alpha21 = np.exp(logalpha21)

        if not self.fit_gamma:
            return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21

        gamma12 = np.exp(loggamma12)
        gamma21 = np.exp(loggamma21)
        return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21

    def _transform_params_to_fit(self, params):
        """Transform appropriate linear params to log-scale for fitting"""
        if not self.fit_gamma:
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21 = params
        else:
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21 = params

        with np.errstate(divide="ignore"):
            logh1 = np.log(h1)
            logh2 = np.log(h2)
            logC1 = np.log(C1)
            logC2 = np.log(C2)
            logalpha12 = np.log(alpha12)
            logalpha21 = np.log(alpha21)

        if not self.fit_gamma:
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21

        with np.errstate(divide="ignore"):
            loggamma12 = np.log(gamma12)
            loggamma21 = np.log(gamma21)
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

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, r1r, r2r, alpha12, alpha21, gamma12, gamma21):
        # Precompute some terms that are used repeatedly
        d1_pow_h1 = np.float_power(d1, h1)
        d2_pow_h2 = np.float_power(d2, h2)
        C1_pow_h1 = np.float_power(C1, h1)
        C2_pow_h2 = np.float_power(C2, h2)

        r1 = r1r / C1_pow_h1
        r2 = r2r / C2_pow_h2

        alpha21_d1_pow_gamma21_h1 = np.float_power(alpha21 * d1, gamma21 * h1)
        alpha12_d2_pow_gamma12_h2 = np.float_power(alpha12 * d2, gamma12 * h2)
        r1_C1h1_pow_gamma21 = np.float_power((r1 * C1_pow_h1), gamma21)
        r2_C2h2_pow_gamma12 = np.float_power((r2 * C2_pow_h2), gamma12)
        r1_pow_gamma21_plus_1 = np.float_power(r1, (gamma21 + 1))
        r2_pow_gamma12_plus_1 = np.float_power(r2, (gamma12 + 1))
        r1_pow_gamma21 = np.float_power(r1, gamma21)
        r2_pow_gamma12 = np.float_power(r2, gamma12)

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

    def _bootstrap_resample(self, d1, d2, E, use_jacobian, bootstrap_iterations, **kwargs):
        super()._bootstrap_resample(d1, d2, E, use_jacobian, bootstrap_iterations, **kwargs)
        self.bootstrap_beta = None
        if self.bootstrap_parameters is None:
            return

        params = self._parameter_names
        E0 = self.bootstrap_parameters[:, params.index("E0")]  # type: ignore
        E1 = self.bootstrap_parameters[:, params.index("E1")]  # type: ignore
        E2 = self.bootstrap_parameters[:, params.index("E2")]  # type: ignore
        E3 = self.bootstrap_parameters[:, params.index("E3")]  # type: ignore
        self.bootstrap_beta = MuSyC._get_beta(E0, E1, E2, E3)

    def get_confidence_intervals(self, confidence_interval: float = 95):
        """Returns the lower bound and upper bound estimate for each parameter.

        Parameters:
        -----------
        confidence_interval : float, default=95
            % confidence interval to return. Must be between 0 and 100.
        """
        ci = super().get_confidence_intervals(confidence_interval=confidence_interval)

        lb = (100 - confidence_interval) / 2.0
        ub = 100 - lb
        ci["beta"] = np.percentile(self.bootstrap_beta, [lb, ub])
        return ci

    def summarize(self, confidence_interval: float = 95, tol: float = 0.01):
        """-"""
        pars = self.get_parameters()

        header = ["Parameter", "Value", "Comparison", "Synergy"]
        ci: dict[str, tuple[float, float]] = {}
        if self.bootstrap_parameters is not None:
            ci = self.get_confidence_intervals(confidence_interval=confidence_interval)
            header.insert(2, f"{confidence_interval:0.3g}% CI")

        rows = [header]

        # beta
        rows.append(self._make_summary_row("beta", 0, self.beta, ci, tol, False, "synergistic", "antagonistic"))

        # alpha and gamma
        for key in pars.keys():
            if "alpha" in key or "gamma" in key:
                rows.append(self._make_summary_row(key, 1, pars[key], ci, tol, True, "synergistic", "antagonistic"))

        print(format_table(rows))

    def __repr__(self):
        if self.is_specified:
            parameters = self.get_parameters()
            parameters["beta"] = self.beta
            param_vals = ", ".join([f"{param}={val:0.3g}" for param, val in parameters.items()])  # typing: ignore
        else:
            param_vals = ""
        return f"MuSyC({param_vals})"

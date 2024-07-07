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

from typing import Dict, Tuple, Type

import numpy as np

from synergy.combination.synergy_model_2d import ParametricSynergyModel2D
from synergy.exceptions import ModelNotParameterizedError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill
from synergy.utils import format_table
from synergy.utils.model_mixins import ParametricModelMixins


class BRAID(ParametricSynergyModel2D):
    """BRAID synergy.

    kappa and delta are the BRAID synergy parameters, though E3 is related to how much more effective the combination is
    than either drug alone. Note though that lim_{d1 -> inf, d2 -> inf}E(d1, d2) does not equal E3 in BRAID.

    .. csv-table:: Interpretation of synergy parameters
       :header: "Parameter", "Values", "Synergy/Antagonism"

       "``kappa``", "< 0",    "Antagonism"
       ,            "> 0",    "Synergism"
       "``delta``", "[0, 1)", "Antagonism"
       ,            "> 1",    "Synergism"

    Parameters
    ----------
    drug1_model : DoseResponseModel1D
        The model for the first drug.

    drug2_model : DoseResponseModel1D
        The model for the second drug.

    mode : str , default="kappa"
        Options are "kappa", "delta", "both". BRAID has model versions that fit synergy using the parameter "kappa", the
        parameter "delta", or both. The standard version only fits kappa, but the other variants are available.
    """

    def __init__(self, drug1_model=None, drug2_model=None, mode="kappa", **kwargs):
        """Ctor."""
        self.mode = mode
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)

        if mode == "kappa":
            self.fit_function = self._model_to_fit_kappa

        elif mode == "delta":
            self.fit_function = self._model_to_fit_delta

        elif mode == "both":
            self.fit_function = self._model_to_fit_both

        self.jacobian_function = None  # TODO

    @property
    def _parameter_names(self):
        params = ["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2"]
        if self.mode in ["kappa", "both"]:
            params.append("kappa")
        if self.mode in ["delta", "both"]:
            params.append("delta")
        return params

    @property
    def _default_fit_bounds(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {
            "h1": (0, np.inf),
            "h2": (0, np.inf),
            "C1": (0, np.inf),
            "C2": (0, np.inf),
        }
        if self.mode in ["kappa", "both"]:
            bounds["kappa"] = (-2, np.inf)
        if self.mode in ["delta", "both"]:
            bounds["delta"] = (0, np.inf)
        return bounds

    def _model_to_fit_kappa(self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa):
        return self._model(
            d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), kappa, 1
        )

    def _model_to_fit_delta(self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logdelta):
        return self._model(
            d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), 0, np.exp(logdelta)
        )

    def _model_to_fit_both(self, d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta):
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
            kappa,
            np.exp(logdelta),
        )

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill

    @property
    def _default_drug1_kwargs(self) -> dict:
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

            if self.mode == "kappa":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0]
            elif self.mode == "delta":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 1]
            else:
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0, 1]

        return super()._get_initial_guess(d1, d2, E, p0)

    def _transform_params_from_fit(self, params):
        if self.mode == "kappa":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa = params
        elif self.mode == "delta":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logdelta = params
            delta = np.exp(logdelta)
        else:
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta = params
            delta = np.exp(logdelta)

        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)

        if self.mode == "kappa":
            return E0, E1, E2, E3, h1, h2, C1, C2, kappa
        elif self.mode == "delta":
            return E0, E1, E2, E3, h1, h2, C1, C2, delta
        return E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta

    def _transform_params_to_fit(self, params):
        with np.errstate(divide="ignore"):
            if self.mode == "kappa":
                E0, E1, E2, E3, h1, h2, C1, C2, kappa = params
            elif self.mode == "delta":
                E0, E1, E2, E3, h1, h2, C1, C2, delta = params
                logdelta = np.log(delta)
            else:
                E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta = params
                logdelta = np.log(delta)

            logh1 = np.log(h1)
            logh2 = np.log(h2)
            logC1 = np.log(C1)
            logC2 = np.log(C2)

        if self.mode == "kappa":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa
        elif self.mode == "delta":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logdelta
        return E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta

    def _set_parameters(self, popt):
        if self.mode == "kappa":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa = popt
        elif self.mode == "delta":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta = popt
        else:
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta = popt

        # E3 is, by definition, the value of E that gives the greatest delta_E. During fitting, the max() function can
        # make E3 have no impact, leading to very sloppy output. Thus here we correct it by setting E3 to whichever E
        # gives the greatest delta_E.

        delta_Es = [self.E1 - self.E0, self.E2 - self.E0, self.E3 - self.E0]
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]
        self.E3 = max_delta_E + self.E0

    def E(self, d1, d2):
        if not self.is_specified:
            raise ModelNotParameterizedError()

        if self.mode == "kappa":
            return self._model(
                d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, 1
            )
        elif self.mode == "delta":
            return self._model(
                d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, 0, self.delta
            )
        return self._model(
            d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta
        )

    def E_reference(self, d1, d2):
        if not self.is_specified:
            raise ModelNotParameterizedError()
        return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, 0, 1)

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta):
        """Model for BRAID.

        See the original in the From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)

        The parameters of this equation must satisfy
          h1>0,
          h2>0,
          delta>0,
          kappa>-2,
          sign(E3-E0)=sign(E1-E0)=sign(E2-E0),
          |E3-E0|>=|E1-E0|, and
          |E3-E0|>=|E2-E0|.
        """
        delta_Es = [E1 - E0, E2 - E0, E3 - E0]
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]

        h = np.sqrt(h1 * h2)
        power = 1 / (delta * h)

        D1 = (
            (E1 - E0)
            / max_delta_E
            * np.float_power(d1 / C1, h1)
            / (1 + (1 - (E1 - E0) / max_delta_E) * np.float_power(d1 / C1, h1))
        )

        D2 = (
            (E2 - E0)
            / max_delta_E
            * np.float_power(d2 / C2, h2)
            / (1 + (1 - (E2 - E0) / max_delta_E) * np.float_power(d2 / C2, h2))
        )

        D = (
            np.float_power(D1, power)
            + np.float_power(D2, power)
            + kappa * np.sqrt(np.float_power(D1, power) * np.float_power(D2, power))
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return E0 + max_delta_E / (1 + np.float_power(D, -delta * h))

    def _get_parameters(self):
        if self.mode == "kappa":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa
        elif self.mode == "delta":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta
        elif self.mode == "both":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta

    def summarize(self, confidence_interval: float = 95, tol: float = 0.01):
        pars = self.get_parameters()

        header = ["Parameter", "Value", "Comparison", "Synergy"]
        ci: Dict[str, Tuple[float, float]] = {}
        if self.bootstrap_parameters is not None:
            ci = self.get_confidence_intervals(confidence_interval=confidence_interval)
            header.insert(2, f"{confidence_interval:0.3g}% CI")

        rows = [header]

        for key in pars.keys():
            if key == "kappa":
                rows.append(
                    ParametricModelMixins.make_summary_row(
                        key, 0, pars[key], ci, tol, False, "synergistic", "antagonistic"
                    )
                )
            elif key == "delta":
                rows.append(
                    ParametricModelMixins.make_summary_row(
                        key, 1, pars[key], ci, tol, True, "synergistic", "antagonistic"
                    )
                )

        print(format_table(rows))

    def __repr__(self):
        if not self.is_specified:
            return "BRAID()"

        if self.mode == "kappa":
            return (
                "BRAID(E0=%0.3g, E1=%0.3g, E2=%0.3g, E3=%0.3g, h1=%0.3g, h2=%0.3g, C1=%0.3g, C2=%0.3g, kappa=%0.3g)"
                % (self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa)
            )
        elif self.mode == "delta":
            return (
                "BRAID(E0=%0.3g, E1=%0.3g, E2=%0.3g, E3=%0.3g, h1=%0.3g, h2=%0.3g, C1=%0.3g, C2=%0.3g, delta=%0.3g)"
                % (self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta)
            )
        return (
            "BRAID(E0=%0.3g, E1=%0.3g, E2=%0.3g, E3=%0.3g, h1=%0.3g, h2=%0.3g, C1=%0.3g, C2=%0.3g, kappa=%0.3g, "
            "delta=%0.3g)"
            % (self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta)
        )

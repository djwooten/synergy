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

from typing import Dict, List, Tuple, Type

import numpy as np

from synergy.combination.synergy_model_2d import ParametricSynergyModel2D
from synergy.exceptions import ModelNotParameterizedError
from synergy.single import Hill_2P
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.utils import format_table
from synergy.utils.model_mixins import ParametricModelMixins


class Zimmer(ParametricSynergyModel2D):
    """The Effective Dose synergy model from Zimmer et al.

    This model uses the multiplicative survival principle (i.e., Bliss), but adds a parameter for each drug describing
    how it affects the potency of the other.

    Synergy by Zimmer is described by these parameters

    .. csv-table:: Interpretation of synergy parameters
       :header: "Parameter", "Values", "Synergy/Antagonism", "Interpretation"

       "``a12``", "< 0", "Synergism",  "Drug 2 increases the effective dose (potency) of drug 1"
       ,          "> 0", "Antagonism", "Drug 2 decreases the effective dose (potency) of drug 1"
       "``a21``", "< 0", "Synergism",  "Drug 1 increases the effective dose (potency) of drug 2"
       ,          "> 0", "Antagonism", "Drug 1 decreases the effective dose (potency) of drug 2"
    """

    def __init__(self, drug1_model=None, drug2_model=None, **kwargs):
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)
        self.fit_function = self._model_to_fit
        self.jacobian_function = None  # TODO

    @property
    def _parameter_names(self) -> List[str]:
        return ["h1", "h2", "C1", "C2", "a12", "a21"]

    @property
    def _default_fit_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {"h1": (0, np.inf), "h2": (0, np.inf), "C1": (0, np.inf), "C2": (0, np.inf)}

    def _model_to_fit(self, d, logh1, logh2, logC1, logC2, a12, a21):
        return self._model(d[0], d[1], np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), a12, a21)

    def _get_initial_guess(self, d1, d2, E, p0):
        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:
            drug1 = self.drug1_model
            drug2 = self.drug2_model

            if not (isinstance(drug1, Hill_2P) and isinstance(drug2, Hill_2P)):
                raise ValueError("Wrong single drug types")

            # Fit the single drug models if they were not pre-specified by the user
            if not drug1.is_specified:
                mask = np.where(d2 == min(d2))
                drug1.fit(d1[mask], E[mask])
            if not drug2.is_specified:
                mask = np.where(d1 == min(d1))
                drug2.fit(d2[mask], E[mask])

            # Get initial guesses of h1, h2, C1, and C2 from single-drug fits
            h1, C1 = drug1.h, drug1.C
            h2, C2 = drug2.h, drug2.C

            p0 = [h1, h2, C1, C2, 0, 0]

        return super()._get_initial_guess(d1, d2, E, p0)

    def _transform_params_from_fit(self, params):
        logh1, logh2, logC1, logC2, a12, a21 = params
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        return h1, h2, C1, C2, a12, a21

    def _transform_params_to_fit(self, params):
        h1, h2, C1, C2, a12, a21 = params

        with np.errstate(divide="ignore"):
            logh1 = np.log(h1)
            logh2 = np.log(h2)
            logC1 = np.log(C1)
            logC2 = np.log(C2)

        return logh1, logh2, logC1, logC2, a12, a21

    def _set_parameters(self, popt):
        self.h1, self.h2, self.C1, self.C2, self.a12, self.a21 = popt

    def E(self, d1, d2):
        if not self.is_specified:
            return ModelNotParameterizedError("Must specify the model before calculating E")
        return self._model(d1, d2, self.h1, self.h2, self.C1, self.C2, self.a12, self.a21)

    def E_reference(self, d1, d2):
        if not self.is_specified:
            return ModelNotParameterizedError("Must specify the model before calculating E")
        return self._model(d1, d2, self.h1, self.h2, self.C1, self.C2, 0, 0)

    def _model(self, d1, d2, h1, h2, C1, C2, a12, a21):
        A = d2 + C2 * (a21 + 1) + d2 * a12
        B = d2 * C1 + C1 * C2 + a12 * d2 * C1 - d1 * (d2 + C2 * (a21 + 1))
        C = -d1 * (d2 * C1 + C1 * C2)

        d1p = (-B + np.sqrt(np.float_power(B, 2.0) - 4 * A * C)) / (2.0 * A)
        with np.errstate(divide="ignore"):
            d2p = d2 / (1.0 + a21 / (1.0 + C1 / d1p))

        return (1 - np.float_power(d1p, h1) / (np.float_power(C1, h1) + np.float_power(d1p, h1))) * (
            1 - np.float_power(d2p, h2) / (np.float_power(C2, h2) + np.float_power(d2p, h2))
        )

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill_2P

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill_2P

    @property
    def _default_drug1_kwargs(self) -> dict:
        lb, ub = self._bounds
        param_names = self._parameter_names
        h_idx = param_names.index("h1")
        C_idx = param_names.index("C1")
        return {
            "E0": 1.0,
            "Emax": 0.0,
            "h_bounds": (np.exp(lb[h_idx]), np.exp(ub[h_idx])),
            "C_bounds": (np.exp(lb[C_idx]), np.exp(ub[C_idx])),
        }

    @property
    def _default_drug2_kwargs(self) -> dict:
        lb, ub = self._bounds
        param_names = self._parameter_names
        h_idx = param_names.index("h2")
        C_idx = param_names.index("C2")
        return {
            "E0": 1.0,
            "Emax": 0.0,
            "h_bounds": (np.exp(lb[h_idx]), np.exp(ub[h_idx])),
            "C_bounds": (np.exp(lb[C_idx]), np.exp(ub[C_idx])),
        }

    def summarize(self, confidence_interval: float = 95, tol: float = 0.01):
        pars = self.get_parameters()

        header = ["Parameter", "Value", "Comparison", "Synergy"]
        ci: Dict[str, Tuple[float, float]] = {}
        if self.bootstrap_parameters is not None:
            ci = self.get_confidence_intervals(confidence_interval=confidence_interval)
            header.insert(2, f"{confidence_interval:0.3g}% CI")

        rows = [header]

        for key in pars.keys():
            if key in ["a12", "a21"]:
                rows.append(
                    ParametricModelMixins.make_summary_row(
                        key, 0, pars[key], ci, tol, False, "antagonistic", "synergistic"
                    )
                )

        print(format_table(rows))

    def __repr__(self):
        if not self.is_specified:
            return "Zimmer()"
        return "Zimmer(h1=%0.3g, h2=%0.3g, C1=%0.3g, C2=%0.3g, a12=%0.3g, a21=%0.3g)" % (
            self.h1,
            self.h2,
            self.C1,
            self.C2,
            self.a12,
            self.a21,
        )

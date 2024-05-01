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

from synergy.exceptions import InvalidDrugModelError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill
from synergy.higher.synergy_model_Nd import DoseDependentSynergyModelND


class Schindler(DoseDependentSynergyModelND):
    """Schindler's multidimensional Hill equation model.

    From "Theory of synergistic effects: Hill-type response surfaces as 'null-interaction' models for mixtures" - Michael Schindler. This model is built to satisfy the Loewe additivity criterion,

    Schindler assumes each drug has a Hill dose response, and defines a multidimensional Hill equation that satisfies Loewe additivity. Unlike Loewe, Schindler can be defined for combinations whose effect exceeds Emax of either drug (Loewe is limited by Emax of the *weaker* drug).

    Synergy is simply defined as the difference between the observed E and the E predicted by Schindler's multidimensional Hill equation.

    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """

    def E_Reference(self, d):
        """-"""
        if not self.is_specified:
            raise InvalidDrugModelError("Model is not specified.")

        E0 = 0
        for single in self.single_drug_models:
            E0 += single.E0 / self.N

        with np.errstate(divide="ignore", invalid="ignore"):
            # Schindler assumes drugs start at 0 and go up to Emax
            self.reference = self._model(d, E0)

    def _get_synergy(self, d, E):
        """-"""
        E0 = 0
        for single in self.single_drug_models:
            E0 += single.E0 / self.N
        uE = E0 - E
        return self._sanitize_synergy(d, uE - self.reference, 0)

    def _model(self, d, E0):
        """
        From "Theory of synergistic effects: Hill-type response surfaces as 'null-interaction' models for mixtures" - Michael Schindler

        E - u_hill = 0 : Additive
        E - u_hill > 0 : Synergistic
        E - u_hill < 0 : Antagonistic
        """
        h = np.asarray([model.h for model in self.single_drug_models])
        C = np.asarray([model.C for model in self.single_drug_models])
        Emax = E0 - np.asarray([model.Emax for model in self.single_drug_models])

        m = d / C

        y = (h * m).sum(axis=1) / m.sum(axis=1)
        u_max = (Emax * m).sum(axis=1) / m.sum(axis=1)
        power = np.power(m.sum(axis=1), y)

        return u_max * power / (1.0 + power)

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return Hill

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return Hill

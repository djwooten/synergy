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
from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.exceptions import InvalidDrugModelError
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.log_linear import LogLinear
from synergy.synergy_exceptions import ModelNotParameterizedError


class Bliss(DoseDependentSynergyModel2D):
    """Bliss independence model

    Bliss synergy is defined as the difference between the observed E and the E predicted by the Bliss Independence
    assumption:
        E_pred = E_drug1_alone * E_drug2_alone

    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """

    def E_reference(self, d1, d2):
        if not self.is_specified:
            raise InvalidDrugModelError("Model is not specified.")
        E1_alone = self.drug1_model.E(d1)
        E2_alone = self.drug2_model.E(d2)

        return E1_alone * E2_alone

    @property
    def synergy_threshold(self) -> float:
        """TODO"""
        return 0.0

    def get_synergy_status(self, tol: float = 0):
        """TODO"""
        if self.synergy is None:
            raise ModelNotParameterizedError("Model has not been fit to data. Call model.fit(d1, d2, E) first.")
        status = np.asarray(["Additive"] * len(self.synergy), dtype=object)
        status[self.synergy < self.synergy_threshold - tol] = "Antagonistic"
        status[self.synergy > self.synergy_threshold + tol] = "Synergistic"
        status[np.where(np.isnan(self.synergy))] = ""
        return status

    def _get_synergy(self, d1, d2, E):
        synergy = self.reference - E
        return self._sanitize_synergy(d1, d2, synergy, 0)

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return DoseResponseModel1D

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return LogLinear

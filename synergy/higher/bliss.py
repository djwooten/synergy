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

from typing import Type

from synergy.exceptions import InvalidDrugModelError
from synergy.higher.synergy_model_Nd import DoseDependentSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.log_linear import LogLinear


class Bliss(DoseDependentSynergyModelND):
    """Bliss independence synergy model for n-drug combinations."""

    def E_reference(self, d):
        if not self.is_specified:
            raise InvalidDrugModelError("Model is not specified.")
        E = 0 * d[:, 0] + 1  # Initialize to 1
        for i, model in enumerate(self.single_drug_models):
            E *= model.E(d[:, i])
        return E

    def _get_synergy(self, d, E):
        synergy = self.reference - E
        return self._sanitize_synergy(d, synergy, 0)

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return DoseResponseModel1D

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return LogLinear

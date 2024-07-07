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

import numpy as np

from synergy.higher.synergy_model_Nd import DoseDependentSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.log_linear import LogLinear


class Loewe(DoseDependentSynergyModelND):
    """The Loewe Additivity synergy model."""

    def E_reference(self, d):
        # TODO: Implement multidimensional Loewe reference using the quadratic minimization in the 2D version
        return d * np.nan

    def _get_synergy(self, d, E):
        d_singles = d * 0  # The dose of each drug that alone achieves E
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(self.N):
                d_singles[:, i] = self.single_drug_models[i].E_inv(E)

            synergy = (d / d_singles).sum(axis=1)
            return self._sanitize_synergy(d, synergy, 1)

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return DoseResponseModel1D

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return LogLinear

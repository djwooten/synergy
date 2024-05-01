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

from synergy.higher.synergy_model_Nd import DoseDependentSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.log_linear import LogLinear


class Loewe(DoseDependentSynergyModelND):
    """Loewe Additivity Synergy

    Loewe's model of drug combination additivity expects a linear tradeoff, such that withholding X parts of drug 1 can be compensated for by adding Y parts of drug 2. In Loewe's model, X and Y are constant (e.g., withholding 5X parts of drug 1 will be compensated for with 5Y parts of drug 2.)

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
    """

    def E_reference(self, d):
        """-"""
        return d * np.nan

    def _get_synergy(self, d, E):
        """-"""
        d_singles = d * 0  # The dose of each drug that alone achieves E
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(self.N):
                d_singles[:, i] = self.single_drug_models[i].E_inv(E)

            synergy = (d / d_singles).sum(axis=1)
            return self._sanitize_synergy(d, synergy, 1)

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return DoseResponseModel1D

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return LogLinear

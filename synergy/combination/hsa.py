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

from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.log_linear import LogLinear


class HSA(DoseDependentSynergyModel2D):
    """Highest single agent (HSA)

    HSA says that any improvement a combination gives over the strongest single agent counts as synergy.

    Members
    -------
    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """

    def __init__(self, stronger_orientation=np.minimum, drug1_model=None, drug2_model=None, **kwargs):
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)
        self.stronger_orientation = stronger_orientation

    def E_reference(self, d1, d2):
        E1_alone = self.drug1_model.E(d1)
        E2_alone = self.drug2_model.E(d2)

        return self.stronger_orientation(E1_alone, E2_alone)

    def _get_synergy(self, d1, d2, E):
        synergy = self.reference - E
        return self._sanitize_synergy(d1, d2, synergy, 0)

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return DoseResponseModel1D

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return LogLinear

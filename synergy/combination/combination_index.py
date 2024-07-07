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

from synergy.combination.schindler import Schindler
from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill_CI


class CombinationIndex(DoseDependentSynergyModel2D):
    """The Combination Index (CI) model of drug synergy.

    Members
    -------
    synergy : array_like, float
        (0,1)=synergism, (1,inf)=antagonism
    """

    def E_reference(self, d1, d2):
        """Use the Schindler model to calculate a reference response."""
        reference_model = Schindler(drug1_model=self.drug1_model, drug2_model=self.drug2_model)
        return reference_model.E_reference(d1, d2)

    def _get_synergy(self, d1, d2, E):
        """Calculate CI.

        CI = d1 / E_inv1(E) + d2 / E_inv2(E)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            d1_alone = self.drug1_model.E_inv(E)
            d2_alone = self.drug2_model.E_inv(E)
            synergy = d1 / d1_alone + d2 / d2_alone

        return self._sanitize_synergy(d1, d2, synergy, 1.0)

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        """The default drug model to use"""
        return Hill_CI

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        """The required superclass of the models for the individual drugs, or None if any model is acceptable"""
        return Hill_CI

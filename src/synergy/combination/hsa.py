"""
    Copyright (C) 2020 David J. Wooten

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from .. import utils
from ..single import MarginalLinear
from .base import DoseDependentModel

class HSA(DoseDependentModel):
    """
    """

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):
        
        super().fit(d1, d2, E)

        #self.synergy = []
        d1_min = np.min(d1)
        d2_min = np.min(d2)

        if (d1_min > 0 or d2_min > 0):
            print("WARNING: HSA expects single-drug information")
        
        d1_alone_mask = np.where(d2==d2_min)
        d2_alone_mask = np.where(d1==d1_min)

        if drug1_model is None:
            drug1_model = MarginalLinear(d=d1[d1_alone_mask], E=E[d1_alone_mask])

        if drug2_model is None:
            drug2_model = MarginalLinear(d=d2[d2_alone_mask], E=E[d2_alone_mask])

        self.drug1_model = drug1_model
        self.drug2_model = drug2_model

        E1_alone = drug1_model.E(d1)
        E2_alone = drug2_model.E(d2)
        self.synergy = np.minimum(E1_alone-E, E2_alone-E)
        
        self.synergy[d1_alone_mask] = 0
        self.synergy[d2_alone_mask] = 0
        return self.synergy
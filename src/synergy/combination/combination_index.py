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
from ..single import hill as hill
from .base import DoseDependentModel

class CombinationIndex(DoseDependentModel):
    """The Combination Index (CI) is a mass-action-derived model of drug combination synergy. In the original 1984 paper (XXX), CI was derived with two forms: (1) mutually-exclusive drugs, and (2) mutually non-exclustive drugs. Since then, model (1) has been preferred, and is the one implemented here.

    CI fits single-drug responses using a log-linearization scatchard-like regression, that implicitly assumes E0=1 and Emax=0, so should only be applied in those cases. If this assumption is not met, users may wish to use Loewe(), which is here calculated identically to CI, but without the E0 and Emax assumptions, and fits drugs to a 4-parameter Hill equation using nonlinear optimization.    

    Synergy Parameters
    ------------------

    synergy : array-like, [0,1)=synergism, (1,inf)=antagonism
    """

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):

        super().fit(d1,d2,E)

        if drug1_model is None or not isinstance(drug1_model, hill.Hill_CI):
            mask = np.where(d2==min(d2))
            drug1_model = hill.Hill_CI.create_fit(d1[mask], E[mask])
        if drug2_model is None or not isinstance(drug2_model, hill.Hill_CI):
            mask = np.where(d1==min(d1))
            drug2_model = hill.Hill_CI.create_fit(d2[mask], E[mask])
        
        self.drug1_model = drug1_model
        self.drug2_model = drug2_model

        with np.errstate(divide='ignore', invalid='ignore'):
            d1_alone = drug1_model.E_inv(E)
            d2_alone = drug2_model.E_inv(E)

            self.synergy = d1/d1_alone + d2/d2_alone

        self.synergy[(d1==0) | (d2==0)] = 1
        
        return self.synergy
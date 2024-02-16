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

from ..single import MarginalLinear
from .. import utils
from .nonparametric_base import DoseDependentModel

class Bliss(DoseDependentModel):
    """Bliss independence model
    
    Bliss synergy is defined as the difference between the observed E and the E predicted by the Bliss Independence assumption (E_pred = E_drug1_alone * E_drug2_alone).

    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """
    
    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)
        super().fit(d1,d2,E, drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)

        drug1_model = self.drug1_model
        drug2_model = self.drug2_model

        E1_alone = drug1_model.E(d1)
        E2_alone = drug2_model.E(d2)

        self.reference = E1_alone*E2_alone
        self.synergy = self.reference - E
        self.synergy[(d1==0) | (d2==0)] = 0

        return self.synergy

    def _get_single_drug_classes(self):
        return MarginalLinear, None
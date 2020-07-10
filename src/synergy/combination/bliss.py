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

        self.synergy = E1_alone*E2_alone - E
        self.synergy[(d1==0) | (d2==0)] = 0

        return self.synergy

    def null_E(self, d1, d2, drug1_model=None, drug2_model=None):
        """Returns E from the Bliss null model (E = E1*E2)
        """
        if self.drug1_model is None or drug1_model is not None: self.drug1_model = drug1_model

        if self.drug2_model is None or drug2_model is not None: self.drug2_model = drug2_model

        if None in [self.drug1_model, self.drug2_model]:
            # Raise model not set error
            ret = 0*d1
            ret[:] = np.nan
            return ret

        D1, D2 = np.meshgrid(d1, d2)
        D1 = D1.flatten()
        D2 = D2.flatten()

        E1_alone = self.drug1_model.E(D1)
        E2_alone = self.drug2_model.E(D2)

        return D1, D2, E1_alone*E2_alone

    def _get_single_drug_classes(self):
        return MarginalLinear, None
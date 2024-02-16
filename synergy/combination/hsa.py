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
import warnings

from .. import utils
from ..single import MarginalLinear
from .nonparametric_base import DoseDependentModel


class HSA(DoseDependentModel):
    """Highest single agent (HSA)

    HSA says that any improvement a combination gives over the strongest single agent counts as synergy.
    """

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):
        
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)
        super().fit(d1,d2,E, drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)

        drug1_model = self.drug1_model
        drug2_model = self.drug2_model

        #self.synergy = []
        d1_min = np.min(d1)
        d2_min = np.min(d2)

        if (d1_min > 0 or d2_min > 0):
            warnings.warn("WARNING: HSA expects single-drug information for both drugs. min(d1)=%0.2e, min(d2)=%0.2e"%(d1_min, d2_min))
        

        self.drug1_model = drug1_model
        self.drug2_model = drug2_model

        E1_alone = drug1_model.E(d1)
        E2_alone = drug2_model.E(d2)

        self.reference = np.minimum(E1_alone, E2_alone)
        #self.synergy = np.minimum(E1_alone-E, E2_alone-E)
        self.synergy = self.reference - E
        
        self.synergy[(d1==0) | (d2==0)] = 0
        return self.synergy

    def _get_single_drug_classes(self):
        return MarginalLinear, None
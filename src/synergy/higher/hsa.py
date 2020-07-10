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
from .nonparametric_base import DoseDependentHigher

class HSA(DoseDependentHigher):
    """Highest single agent (HSA)

    HSA says that any improvement a combination gives over the strongest single agent counts as synergy.
    """
    
    def fit(self, d, E, single_models=None, **kwargs):
        d = np.asarray(d)
        E = np.asarray(E)
        n = d.shape[1]

        super().fit(d, E, single_models=single_models, **kwargs)
        
        # Fit single drugs
        if single_models is None:
            single_models = []
            
            for i in range(n):
                # Mask where all other drugs are minimum (ideally 0)
                mask = d[:,i]>=0 # This should always be true
                for j in range(n):
                    if i==j: continue
                    mask = mask & (d[:,j]==np.min(d[:,j]))
                mask = np.where(mask)
                single = MarginalLinear()
                single.fit(d[mask,i].flatten(), E[mask], **kwargs)
                single_models.append(single)
        self.single_models = single_models

        E_singles = d*0
        for i in range(n):
            E_singles[:,i] = single_models[i].E(d[:,i])
        E_HSA = np.min(E_singles, axis=1)

        # Calculate synergy as excess over HSA
        self.synergy = E_HSA - E

        # Ensure all single-drug HSA scores are 1
        for i in range(n):
                # Mask where all other drugs are minimum (ideally 0)
                mask = d[:,i]>=0 # This should always be true
                for j in range(n):
                    if i==j: continue
                    mask = mask & (d[:,j]==np.min(d[:,j]))
                mask = np.where(mask)
                self.synergy[mask] = 0

        return self.synergy

    def _get_single_drug_classes(self):
        return MarginalLinear, None
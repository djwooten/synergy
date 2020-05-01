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

from ..single import Hill
from .nonparametric_base import DoseDependentHigher

class Bliss(DoseDependentHigher):
    """Bliss independence model
    
    Bliss synergy is defined as the difference between the observed E and the E predicted by the Bliss Independence assumption (E_pred = E_drug1_alone * E_drug2_alone). This is also known as excess over Bliss.

    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """

    def __init__(self, E_bounds=(0,1), h_bounds=(0,np.inf),         \
            C_bounds=(0,np.inf)):
            super().__init__()

            self.E_bounds = E_bounds
            self.h_bounds = h_bounds
            self.C_bounds = C_bounds
    
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
                single = Hill(E0_bounds=self.E_bounds, Emax_bounds=self.E_bounds, h_bounds=self.h_bounds, C_bounds=self.C_bounds)
                single.fit(d[mask,i].flatten(), E[mask], **kwargs)
                single_models.append(single)
        self.single_models = single_models

        # Get E for each single drug
        E_singles = d*0
        for i in range(n):
            E_singles[:,i] = single_models[i].E(d[:,i])
        E_bliss = np.prod(E_singles, axis=1)
        
        # Calculate synergy as excess over bliss
        self.synergy = E_bliss - E

        # Ensure all single-drug bliss scores are 0
        for i in range(n):
                # Mask where all other drugs are minimum (ideally 0)
                mask = d[:,i]>=0 # This should always be true
                for j in range(n):
                    if i==j: continue
                    mask = mask & (d[:,j]==np.min(d[:,j]))
                mask = np.where(mask)
                self.synergy[mask] = 0

        return self.synergy

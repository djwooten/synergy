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

class Loewe(DoseDependentHigher):
    """Loewe Additivity Synergy
    
    Loewe's model of drug combination additivity expects a linear tradeoff, such that withholding X parts of drug 1 can be compensated for by adding Y parts of drug 2. In Loewe's model, X and Y are constant (e.g., withholding 5X parts of drug 1 will be compensated for with 5Y parts of drug 2.)

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
    """
    
    def fit(self, d, E, single_models=None, **kwargs):
        d = np.asarray(d)
        E = np.asarray(E)
        n = d.shape[1]
        
        super().fit(d, E, single_models=single_models, **kwargs)
        single_models=self.single_models

        d_singles = d*0 # The dose of each drug that alone achieves E
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(n):
                d_singles[:,i] = single_models[i].E_inv(E)

            self.synergy = (d/d_singles).sum(axis=1)

        # Ensure all single-drug Loewe scores are 1
        for i in range(n):
                # Mask where all other drugs are minimum (ideally 0)
                mask = d[:,i]>=0 # This should always be true
                for j in range(n):
                    if i==j: continue
                    mask = mask & (d[:,j]==np.min(d[:,j]))
                mask = np.where(mask)
                self.synergy[mask] = 1

        return self.synergy

    def _get_single_drug_classes(self):
        return Hill, None

    def plotly_isosurfaces(self, drug_axes=[0,1,2], other_drug_slices=None, cmap="PRGn", neglog=True, **kwargs):
        super().plotly_isosurfaces(drug_axes=drug_axes, other_drug_slices=other_drug_slices, cmap=cmap, neglog=neglog, **kwargs)
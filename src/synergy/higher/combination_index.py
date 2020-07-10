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

from ..single import Hill_CI
from .nonparametric_base import DoseDependentHigher

class CombinationIndex(DoseDependentHigher):
    """Combination Index (CI).
    
    CI is a mass-action-derived model of drug combination synergy. In the original 1984 paper (XXX), CI was derived with two forms: (1) mutually-exclusive drugs, and (2) mutually non-exclustive drugs. Since then, model (1) has been preferred, and is the one implemented here.

    CI fits single-drug responses using a log-linearization scatchard-like regression, that implicitly assumes E0=1 and Emax=0, so should only be applied in those cases. If this assumption is not met, users may wish to use Loewe(), which is here calculated identically to CI, but without the E0 and Emax assumptions, and fits drugs to a 4-parameter Hill equation using nonlinear optimization.

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
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
                single = Hill_CI.create_fit(d[mask,i].flatten(), E[mask])
                single_models.append(single)
        self.single_models = single_models

        d_singles = d*0
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(n):
                d_singles[:,i] = single_models[i].E_inv(E)

            self.synergy = (d/d_singles).sum(axis=1)

        # Ensure all single-drug CI scores are 1
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
        return Hill_CI, Hill_CI

    def plotly_isosurfaces(self, drug_axes=[0,1,2], other_drug_slices=None, cmap="PRGn", neglog=True, **kwargs):
        super().plotly_isosurfaces(drug_axes=drug_axes, other_drug_slices=other_drug_slices, cmap=cmap, neglog=neglog, **kwargs)
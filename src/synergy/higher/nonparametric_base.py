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
from ..utils import plots


class DoseDependentHigher:
    """These are models for which synergy is defined independently at each individual dose.
    """
    def __init__(self):
        """Creates a DoseDependentModel

        """
        self.synergy = None
        self.d = None
        self.single_models = None

    def fit(self, d, E, single_models=None, **kwargs):
        """Calculates dose-dependent synergy at doses d1, d2.

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2

        E : array_like
            Dose-response at doses d1 and d2

        drug1_model : single-drug-model, default=None
            Pre-defined, or fit, model (e.g., Hill()) of drug 1 alone. If None (default), then d1 and E will be masked where d2==min(d2), and used to fit a model (Hill, Hill_2P, or Hill_CI, depending on the synergy model) for drug 1.

        drug2_model : single-drug-model, default=None
            Same as drug1_model, for drug 2.
        
        kwargs
            kwargs to pass to Hill.fit() (or whichever single-drug model is used)

        Returns
        ----------
        synergy : array_like
            The synergy calculated at all doses d1, d2
        """
        self.d = d
        self.synergy = 0*E
        self.synergy[:] = np.nan
        return self.synergy


    def plotly_isosurfaces(self, drug_axes=[0,1,2], other_drug_slices=None, cmap="PRGn", neglog=False, **kwargs):
        mask = self.d[:,0]>0
        n = self.d.shape[1]
        for i in range(n):
            if i in drug_axes:
                continue
            if other_drug_slices is None:
                dslice = np.min(self.d[:,i])
            else:
                dslice = other_drug_slices[i]
            mask = mask & (self.d[:,i]==dslice)

        d1 = self.d[mask,drug_axes[0]]
        d2 = self.d[mask,drug_axes[1]]
        d3 = self.d[mask,drug_axes[2]]

        Z = self.synergy[mask]
        if neglog:
            Z = -np.log10(Z)

        plots.plotly_isosurfaces(d1, d2, d3, Z, cmap=cmap, center_on_zero=True, **kwargs)
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
from .nonparametric_base import DoseDependentModel

class Loewe(DoseDependentModel):
    """Loewe Additivity Synergy
    
    Loewe's model of drug combination additivity expects a linear tradeoff, such that withholding X parts of drug 1 can be compensated for by adding Y parts of drug 2. In Loewe's model, X and Y are constant (e.g., withholding 5X parts of drug 1 will be compensated for with 5Y parts of drug 2.)

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
    """
    
    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):

        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)
        super().fit(d1,d2,E)

        if drug1_model is None:
            mask = np.where(d2==min(d2))
            drug1_model = Hill(E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)
            drug1_model.fit(d1[mask], E[mask], **kwargs)
        
        if drug2_model is None:
            mask = np.where(d1==min(d1))
            drug2_model = Hill(E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)
            drug2_model.fit(d2[mask], E[mask], **kwargs)
        
        self.drug1_model = drug1_model
        self.drug2_model = drug2_model

        self.synergy = self._get_synergy(d1, d2, E, drug1_model, drug2_model)

        return self.synergy
    
    def _get_synergy(self, d1, d2, E, drug1_model, drug2_model):
        with np.errstate(divide='ignore', invalid='ignore'):
            d1_alone = drug1_model.E_inv(E)
            d2_alone = drug2_model.E_inv(E)
            synergy = d1/d1_alone + d2/d2_alone
        synergy[(d1==0) | (d2==0)] = 1
        return synergy


    def plot_colormap(self, cmap="PRGn", neglog=True, center_on_zero=True, **kwargs):
        super().plot_colormap(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)
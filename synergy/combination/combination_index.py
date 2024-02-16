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
from .nonparametric_base import DoseDependentModel
from .schindler import Schindler

class CombinationIndex(DoseDependentModel):
    """Combination Index (CI).
    
    CI is a mass-action-derived model of drug combination synergy. In the original 1984 paper (XXX), CI was derived with two forms: (1) mutually-exclusive drugs, and (2) mutually non-exclustive drugs. Since then, model (1) has been preferred, and is the one implemented here.

    CI fits single-drug responses using a log-linearization scatchard-like regression, that implicitly assumes E0=1 and Emax=0, so should only be applied in those cases. If this assumption is not met, users may wish to use Loewe(), which is here calculated identically to CI, but without the E0 and Emax assumptions, and fits drugs to a 4-parameter Hill equation using nonlinear optimization.

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
    """

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)
        super().fit(d1,d2,E, drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)

        drug1_model = self.drug1_model
        drug2_model = self.drug2_model

        with np.errstate(divide='ignore', invalid='ignore'):
            d1_alone = drug1_model.E_inv(E)
            d2_alone = drug2_model.E_inv(E)

            self.synergy = d1/d1_alone + d2/d2_alone

        reference_model = Schindler()
        self.reference = 1-reference_model._model(d1, d2, 1, 1, drug1_model.h, drug2_model.h, drug1_model.C, drug2_model.C)

        self.synergy[(d1==0) | (d2==0)] = 1
        
        return self.synergy

    def _get_single_drug_classes(self):
        return Hill_CI, Hill_CI

    def plot_heatmap(self, cmap="PRGn", neglog=True, center_on_zero=True, **kwargs):
        super().plot_heatmap(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", neglog=True, center_on_zero=True, **kwargs):
        super().plot_surface_plotly(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)
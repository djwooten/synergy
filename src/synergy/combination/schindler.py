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
from .. import utils
from .nonparametric_base import DoseDependentModel

class Schindler(DoseDependentModel):
    """Schindler's multidimensional Hill equation model.
    
    From "Theory of synergistic effects: Hill-type response surfaces as 'null-interaction' models for mixtures" - Michael Schindler. This model is built to satisfy the Loewe additivity criterion, 

    Schindler assumes each drug has a Hill dose response, and defines a multidimensional Hill equation that satisfies Loewe additivity. Unlike Loewe, Schindler can be defined for combinations whose effect exceeds Emax of either drug (Loewe is limited by Emax of the *weaker* drug).

    Synergy is simply defined as the difference between the observed E and the E predicted by Schindler's multidimensional Hill equation.

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
        
        E0_1, E1, h1, C1 = self.drug1_model.get_parameters()
        E0_2, E2, h2, C2 = self.drug2_model.get_parameters()
        E0 = (E0_1+E0_2)/2.
        uE1 = E0-E1
        uE2 = E0-E2
        
        uE = E0-E

        # Where d1==0 and d2==0, schindler will divide by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            uE_model = self._model(d1, d2, uE1, uE2, h1, h2, C1, C2)

        self.reference = E0-uE_model
        self.synergy = uE-uE_model

        self.synergy[(d1==0) | (d2==0)] = 0.
        return self.synergy
    

    def _model(self, d1, d2, E1, E2, h1, h2, C1, C2):
        """
        From "Theory of synergistic effects: Hill-type response surfaces as 'null-interaction' models for mixtures" - Michael Schindler
        
        E - u_hill = 0 : Additive
        E - u_hill > 0 : Synergistic
        E - u_hill < 0 : Antagonistic
        """
        m1 = (d1/C1)
        m2 = (d2/C2)
        
        y = (h1*m1 + h2*m2) / (m1+m2)
        u_max = (E1*m1 + E2*m2) / (m1+m2)
        power = np.power(m1+m2, y)
        
        return u_max * power / (1. + power)

    def _get_single_drug_classes(self):
        return Hill, Hill
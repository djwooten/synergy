"""
    Copyright (C) 2020 David J. Wooten

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import synergy.utils.utils as utils
import synergy.single.hill as hill

class Schindler:
    """
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf)):
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds

        self._synergy = None
        self._drug1_model = None
        self._drug2_model = None
        
    
    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None):
        """
        """
        if drug1_model is None:
            mask = np.where(d2==min(d2))
            drug1_model = hill.Hill.create_fit(d1[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)
        if drug2_model is None:
            mask = np.where(d1==min(d1))
            drug2_model = hill.Hill.create_fit(d2[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)
        
        self._drug1_model = drug1_model
        self._drug2_model = drug2_model

        E0_1, E1, h1, C1 = self._drug1_model.get_parameters()
        E0_2, E2, h2, C2 = self._drug2_model.get_parameters()
        E0 = (E0_1+E0_2)/2.
        uE1 = E0-E1
        uE2 = E0-E2
        
        uE = E0-E
        # If d1==0 and d2==0, then schindler will divide by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            uE_model = self._model(d1, d2, uE1, uE2, h1, h2, C1, C2)

        self._synergy = uE-uE_model

        self._synergy[(d1==0) | (d2==0)] = 0.
        return self._synergy
    

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

    #def __repr__(self):
    #    return "Loewe()"
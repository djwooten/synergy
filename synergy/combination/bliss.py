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

class Bliss:
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
        TODO: Add options for fitting ONLY using marginal points, not fits
        """
        if drug1_model is None:
            mask = np.where(d2==min(d2))
            drug1_model = utils.fit_single(d1[mask], E[mask], self.E0_bounds, self.E1_bounds, self.h1_bounds, self.C1_bounds)
        if drug2_model is None:
            mask = np.where(d1==min(d1))
            drug2_model = utils.fit_single(d2[mask], E[mask], self.E0_bounds, self.E2_bounds, self.h2_bounds, self.C2_bounds)
        
        self._drug1_model = drug1_model
        self._drug2_model = drug2_model

        E1_alone = drug1_model.E(d1)
        E2_alone = drug2_model.E(d2)
        self._synergy = E1_alone*E2_alone - E

        return self._synergy

    def null_E(self, d1, d2, drug1_model=None, drug2_model=None):
        if self._drug1_model is None or drug1_model is not None: self._drug1_model = drug1_model

        if self._drug2_model is None or drug2_model is not None: self._drug2_model = drug2_model

        if None in [self._drug1_model, self._drug2_model]:
            # Raise model not set error
            return 0

        D1, D2 = np.meshgrid(d1, d2)
        D1 = D1.flatten()
        D2 = D2.flatten()

        E1_alone = self._drug1_model.E(D1)
        E2_alone = self._drug2_model.E(D2)

        return D1, D2, E1_alone*E2_alone

    
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

class HSA:
    """
    """
    def __init__(self):
        self._synergy = []

    def fit(self, d1, d2, E, **kwargs):
        
        self._synergy = []
        d1_min = np.min(d1)
        d2_min = np.min(d2)

        if (d1_min > 0 or d2_min > 0):
            print("WARNING: HSA expects single-drug information")
        
        for (D1, D2, EX) in zip(d1, d2, E):
            if D1==d1_min or D2==d2_min:
                self._synergy.append(0)
                continue
            d1_alone_mask = np.where((d2==d2_min) & (d1==D1))
            d2_alone_mask = np.where((d1==d1_min) & (d2==D2))
            
            E1_alone = np.mean(E[d1_alone_mask])
            E2_alone = np.mean(E[d2_alone_mask])

            delta_E1 = E1_alone - EX
            delta_E2 = E2_alone - EX

            self._synergy.append(min(delta_E1, delta_E2))

        self._synergy = np.asarray(self._synergy)
        return self._synergy
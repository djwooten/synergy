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

class Schindler(DoseDependentHigher):
    """Schindler's multidimensional Hill equation model.
    
    From "Theory of synergistic effects: Hill-type response surfaces as 'null-interaction' models for mixtures" - Michael Schindler. This model is built to satisfy the Loewe additivity criterion, 

    Schindler assumes each drug has a Hill dose response, and defines a multidimensional Hill equation that satisfies Loewe additivity. Unlike Loewe, Schindler can be defined for combinations whose effect exceeds Emax of either drug (Loewe is limited by Emax of the *weaker* drug).

    Synergy is simply defined as the difference between the observed E and the E predicted by Schindler's multidimensional Hill equation.

    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """
    
    def fit(self, d, E, single_models=None, **kwargs):
        d = np.asarray(d)
        E = np.asarray(E)
        n = d.shape[1]
        super().fit(d, E, single_models=single_models, **kwargs)
        single_models=self.single_models
        E0 = 0
        
        # Fit single drugs
        for single in single_models:
            E0 += single.E0 / n

        with np.errstate(divide='ignore', invalid='ignore'):
            # Schindler assumes drugs start at 0 and go up to Emax
            uE = E0 - E
            uE_schindler = self._model(d, E0)
            self.synergy = uE - uE_schindler

        # Ensure all single-drug Schindler scores are 0
        for i in range(n):
                # Mask where all other drugs are minimum (ideally 0)
                mask = d[:,i]>=0 # This should always be true
                for j in range(n):
                    if i==j: continue
                    mask = mask & (d[:,j]==np.min(d[:,j]))
                mask = np.where(mask)
                self.synergy[mask] = 0

        return self.synergy


    def _model(self, d, E0):
        """
        From "Theory of synergistic effects: Hill-type response surfaces as 'null-interaction' models for mixtures" - Michael Schindler
        
        E - u_hill = 0 : Additive
        E - u_hill > 0 : Synergistic
        E - u_hill < 0 : Antagonistic
        """
        h = np.asarray([model.h for model in self.single_models])
        C = np.asarray([model.C for model in self.single_models])
        Emax = E0 - np.asarray([model.Emax for model in self.single_models])

        m = d/C
        
        y = (h*m).sum(axis=1) / m.sum(axis=1)
        u_max = (Emax*m).sum(axis=1) / m.sum(axis=1)
        power = np.power(m.sum(axis=1), y)
        
        return u_max * power / (1. + power)

    def _get_single_drug_classes(self):
        return Hill, Hill
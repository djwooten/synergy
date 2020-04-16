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
import warnings
from scipy.optimize import curve_fit
from .. import utils
from ..single import hill as hill
from .base import *

class Zimmer(ParameterizedModel):
    """The Effective Dose Model from Zimmer et al (doi: 10.1073/pnas.1606301113). This model uses the multiplicative survival principle (i.e., Bliss), but adds a parameter for each drug describing how it affects the potency of the other. Specifically, given doses d1 and d2, this model translates them to "effective" doses using the following system of equations

                            d1
    d1_eff =  --------------------------------
              1 + a12*(1/(1+(d2_eff/C2)^(-1)))

                            d2
    d2_eff =  --------------------------------
              1 + a21*(1/(1+(d1_eff/C1)^(-1)))

    Synergy Parameters
    ------------------

    a12 : (-(1+(d2_eff/C2))/(d2_eff/C2),0)=synergism, (0,inf)=antagonism
        Describes how drug 2 affects the effective dose of drug 1.

    a21 : (-(1+(d1_eff/C1))/(d1_eff/C1),0)=synergism, (0,inf)=antagonism
        Describes how drug 1 affects the effective dose of drug 2.
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf), a12_bounds=(-np.inf, np.inf), a21_bounds=(-np.inf, np.inf), h1=None, h2=None, C1=None, C2=None, a12=None, a21=None):

        super().__init__()
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.a12_bounds = a12_bounds
        self.a21_bounds = a21_bounds

        with np.errstate(divide='ignore'):
            self.logh1_bounds = (np.log(h1_bounds[0]), np.log(h1_bounds[1]))
            self.logC1_bounds = (np.log(C1_bounds[0]), np.log(C1_bounds[1]))
            self.logh2_bounds = (np.log(h2_bounds[0]), np.log(h2_bounds[1]))
            self.logC2_bounds = (np.log(C2_bounds[0]), np.log(C2_bounds[1]))

        self.h1 = h1
        self.h2 = h2
        self.C1 = C1
        self.C2 = C2
        self.a12 = a12
        self.a21 = a21

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian=True, **kwargs):
        if (use_jacobian):
            warnings.warn(FeatureNotImplemented("Jacobian has not been implemented for Zimmer synergy model, but will still be used for single-drug fits"))
        bounds = tuple(zip(self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.a12_bounds, self.a21_bounds))

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
            for i in range(4):
                p0[i] = np.log(p0[i])
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0
        else:
            if drug1_model is None:
                mask = np.where(d2==min(d2))
                drug1_model = hill.Hill_2P.create_fit(d1[mask], E[mask], h_bounds=self.h1_bounds, C_bounds=self.C1_bounds, use_jacobian=use_jacobian)
                
            if drug2_model is None:
                mask = np.where(d1==min(d1))
                drug2_model = hill.Hill_2P.create_fit(d2[mask], E[mask], h_bounds=self.h2_bounds, C_bounds=self.C2_bounds, use_jacobian=use_jacobian)
            
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            p0 = [h1, h2, C1, C2, 0, 0]
            for i in range(4):
                p0[i] = np.log(p0[i])
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0']  = p0


        xdata = np.vstack((d1,d2))
        
        f = lambda d, logh1, logh2, logC1, logC2, a12, a21: self._model(d[0], d[1], np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), a12, a21)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            popt1, pcov = curve_fit(f, xdata, E, bounds=bounds, **kwargs)

        logh1, logh2, logC1, logC2, a12, a21 = popt1
        self.h1 = np.exp(logh1)
        self.h2 = np.exp(logh2)
        self.C1 = np.exp(logC1)
        self.C2 = np.exp(logC2)
        self.a12 = a12
        self.a21 = a21

        self._score(d1, d2, E)

        return a12, a21

    def E(self, d1, d2):
        if not self._is_parameterized():
            # ERROR
            return 0
        return self._model(d1, d2, self.h1, self.h2, self.C1, self.C2, self.a12, self.a21)

    def _model(self, d1, d2, h1, h2, C1, C2, a12, a21):
        A = d2 + C2*(a21+1) + d2*a12
        B = d2*C1 + C1*C2 + a12*d2*C1 - d1*(d2+C2*(a21+1))
        C = -d1*(d2*C1 + C1*C2)

        d1p = (-B + np.sqrt(np.power(B,2.) - 4*A*C)) / (2.*A)
        d2p = d2 / (1. + a21 / (1. + C1/d1p))
        
        return (1 - d1p**h1/(C1**h1+d1p**h1)) * (1 - d2p**h2/(C2**h2+d2p**h2))

    def get_parameters(self):
        return self.h1, self.h2, self.C1, self.C2, self.a12, self.a21

    def create_fit(d1, d2, E, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf), C1_bounds=(0,np.inf), C2_bounds=(0,np.inf), a12_bounds=(-np.inf,np.inf), a21_bounds=(-np.inf,np.inf), **kwargs):
        model = Zimmer(h1_bounds=h1_bounds, h2_bounds=h2_bounds, C1_bounds=C1_bounds, C2_bounds=C2_bounds, a12_bounds=a12_bounds, a21_bounds=a21_bounds)
        
        model.fit(d, E, **kwargs)
        return model

    def __repr__(self):
        if not self._is_parameterized(): return "Zimmer()"
        return "Zimmer(h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, a12=%0.2f, a21=%0.2f)"%(self.h1, self.h2, self.C1, self.C2, self.a12, self.a21)
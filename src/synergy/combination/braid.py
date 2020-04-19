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

from .. import utils

from ..single import Hill
from .parametric_base import ParametricModel

class BRAID(ParametricModel):
    """BRAID synergy (doi:10.1038/srep25523).
    
    The version implemented here is the "extended" BRAID model with 10 parameters, E0, E1, E2, E3, h1, h2, C1, C2, kappa, and delta.

    kappa and delta are the BRAID synergy parameters, though E3 is related to how much more effective the combination is than either drug alone.
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), E3_bounds=(-np.inf,np.inf), \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf), kappa_bounds=(-np.inf, np.inf), delta_bounds=(0, np.inf), E0=None, E1=None, E2=None, E3=None, h1=None, h2=None, C1=None, C2=None, kappa=None, delta=None):

        super().__init__()
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds
        self.E3_bounds = E3_bounds
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.kappa_bounds = kappa_bounds
        self.delta_bounds = delta_bounds

        with np.errstate(divide='ignore'):
            self.logh1_bounds = (np.log(h1_bounds[0]), np.log(h1_bounds[1]))
            self.logC1_bounds = (np.log(C1_bounds[0]), np.log(C1_bounds[1]))
            self.logh2_bounds = (np.log(h2_bounds[0]), np.log(h2_bounds[1]))
            self.logC2_bounds = (np.log(C2_bounds[0]), np.log(C2_bounds[1]))
            self.logdelta_bounds = (np.log(delta_bounds[0]), np.log(delta_bounds[1]))

        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.h1 = h1
        self.h2 = h2
        self.C1 = C1
        self.C2 = C2
        self.kappa = kappa
        self.delta = delta

        self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), kappa, np.exp(logdelta))

        self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.kappa_bounds, self.logdelta_bounds))


    def _get_initial_guess(self, d1, d2, E, drug1_model=None, drug2_model=None, p0=None):
        
        if p0 is None:
            if drug1_model is None:
                mask = np.where(d2==min(d2))
                drug1_model = Hill.create_fit(d1[mask], E[mask], h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)
            if drug2_model is None:
                mask = np.where(d1==min(d1))
                drug2_model = Hill.create_fit(d2[mask], E[mask], h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)
            
            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            E0 = (E0_1+E0_2)/2
            
            E3 = E[(d1==max(d1)) & (d2==max(d2))]
            if len(E3)>0: E3 = np.mean(E3)
            else: E3 = np.min(E)
                        
            p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0, 1]
            
        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta = params
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        delta = np.exp(logdelta)


        return E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta

    def _transform_params_to_fit(self, params):
        E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta  = params

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)
        logdelta = np.log(delta)
        
        return E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, delta
        

    def _set_parameters(self, popt):
        self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta = popt

    def E(self, d1, d2):
        if not self._is_parameterized():
            # ERROR
            return 0
        return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta)

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta):

        delta_Es = [E1-E0, E2-E0, E3-E0]
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]

        D1 = (E1-E0)/(max_delta_E) * (d1/C1)**h1 / (1+(1-(E1-E0)/(max_delta_E))*(d1/C1)**h1)
        D2 = (E2-E0)/(max_delta_E) * (d2/C2)**h2 / (1+(1-(E2-E0)/(max_delta_E))*(d2/C2)**h2)

        power = 1 / (delta * np.sqrt(h1*h2))
        
        D = D1**power + D2**power + kappa*np.sqrt(D1**power * D2**power)

        return E0 + max_delta_E / (1+D**(-1/power))

    def get_parameters(self):
        return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta

    def create_fit(d1, d2, E, h_bounds=(1e-3,1e3), C_bounds=(0,np.inf),     \
            E_bounds=(-np.inf,np.inf), kappa_bounds=(-1e5,1e5), delta_bounds=(1e-5,1e5), **kwargs):

        model = BRAID(E0_bounds=E_bounds, E1_bounds=E_bounds, E2_bounds=E_bounds, E3_bounds=E_bounds, h1_bounds=h_bounds, h2_bounds=h_bounds, C1_bounds=C_bounds, C2_bounds=C_bounds, kappa_bounds=kappa_bounds, delta_bounds=delta_bounds)
        
        model.fit(d, E, **kwargs)
        return model

    def __repr__(self):
        if not self._is_parameterized(): return "BRAID()"
        return "BRAID(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, kappa=%0.2f, delta=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta)
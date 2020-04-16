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
from scipy.optimize import curve_fit
from . import musyc_jacobian as musyc_jacobian
from .base import *
from .. import utils
from ..single import hill as hill

class MuSyC(ParameterizedModel):
    """Multidimensional Synergy of Combinations (MuSyC) is a drug synergy framework based on the law of mass action (doi: 10.1016/j.cels.2019.01.003, doi: 10.1101/683433). In MuSyC, synergy is parametrically defined as shifts in potency, efficacy, or cooperativity.

    Synergy Parameters
    ------------------

    alpha21 : Synergistic potency ([0,1) = antagonism, (1,inf) = synergism)
        At large concentrations of drug 2, the "effective dose" of drug 1 = alpha21*d1.
    
    alpha12 : Synergistic potency ([0,1) = antagonism, (1,inf) = synergism)
        At large concentrations of drug 1, the "effective dose" of drug 2 = alpha12*d2.

    beta : Synergistic efficacy ((-inf,0) = antagonism, (0,inf) = synergism)
        At large concentrations of both drugs, the combination achieves an effect beta-% stronger (or weaker) than the stronger single-drug.

    NOTE: The current implementation does not yet fit gamma21 or gamma12, but fixes them at 1.

    gamma21 : Synergistic cooperativity ([0,1) = antagonism, (1,inf) = synergism)
        At large concentrations of drug 2, the Hill slope of drug 1 = gamma21*h1

    gamma12 : Synergistic cooperativity ([0,1) = antagonism, (1,inf) = synergism)
        At large concentrations of drug 1, the Hill slope of drug 2 = gamma12*h2

    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), E3_bounds=(-np.inf,np.inf), \
            alpha12_bounds=(0,np.inf), alpha21_bounds=(0,np.inf),   \
            gamma12_bounds=(0,np.inf), gamma21_bounds=(0,np.inf),   \
            r1=1., r2=1., E0=None, E1=None, E2=None, E3=None,   \
            h1=None, h2=None, C1=None, C2=None, alpha12=None,       \
            alpha21=None, gamma12=1., gamma21=1.):
        super().__init__()
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds
        self.E3_bounds = E3_bounds
        self.alpha12_bounds = alpha12_bounds
        self.alpha21_bounds = alpha21_bounds
        self.gamma12_bounds = gamma12_bounds
        self.gamma21_bounds = gamma21_bounds

        self.r1 = r1
        self.r2 = r2
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.h1 = h1
        self.h2 = h2
        self.C1 = C1
        self.C2 = C2
        self.alpha12 = alpha12
        self.alpha21 = alpha21
        self.gamma12 = gamma12
        self.gamma21 = gamma21
        self.beta = None

        self.converged = False

        with np.errstate(divide='ignore'):
            self.logh1_bounds = (np.log(h1_bounds[0]), np.log(h1_bounds[1]))
            self.logC1_bounds = (np.log(C1_bounds[0]), np.log(C1_bounds[1]))
            self.logh2_bounds = (np.log(h2_bounds[0]), np.log(h2_bounds[1]))
            self.logC2_bounds = (np.log(C2_bounds[0]), np.log(C2_bounds[1]))
            
            self.logalpha12_bounds = (np.log(alpha12_bounds[0]), np.log(alpha12_bounds[1]))
            self.logalpha21_bounds = (np.log(alpha21_bounds[0]), np.log(alpha21_bounds[1]))

            self.loggamma12_bounds = (np.log(gamma12_bounds[0]), np.log(gamma12_bounds[1]))
            self.loggamma21_bounds = (np.log(gamma21_bounds[0]), np.log(gamma21_bounds[1]))

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian = True, p0=None, **kwargs):
        
        f = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), self.r1, self.r2, np.exp(logalpha12), np.exp(logalpha21))

        jacobian = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21: musyc_jacobian.jacobian(d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1, self.r2, logalpha12, logalpha21)

        bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logalpha12_bounds, self.logalpha21_bounds))

        p0 = self._get_intial_guess(d1, d2, E, bounds, drug1_model=drug1_model, drug2_model=drug2_model, p0=p0)

        d1 = utils.remove_zeros(d1)
        d2 = utils.remove_zeros(d2)
        xdata = np.vstack((d1,d2))

        try:
            if (use_jacobian):
                popt1, pcov = curve_fit(f, xdata, E, bounds=bounds, jac=jacobian, p0=p0, **kwargs)

            else:
                popt1, pcov = curve_fit(f, xdata, E, bounds=bounds, p0=p0, **kwargs)

            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21 = popt1
            self.converged = True
        except RuntimeError:
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21 = p0
            self.converged = False

        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.h1 = np.exp(logh1)
        self.h2 = np.exp(logh2)
        self.C1 = np.exp(logC1)
        self.C2 = np.exp(logC2)
        self.alpha12 = np.exp(logalpha12)
        self.alpha21 = np.exp(logalpha21)
        self.beta = (min(E1,E2)-E3) / (E0 - min(E1,E2))
        self.gamma12 = 1.
        self.gamma21 = 1.

        if (self._is_parameterized()):
            self._score(d1, d2, E)
            
        return self.alpha12, self.alpha21, self.beta

    def _get_intial_guess(self, d1, d2, E, bounds, drug1_model=None, drug2_model=None, p0=None):
        if p0 is not None:
            p0 = list(p0)
            # p0 = (E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21) - starting from h1 to the end, all guesses must be log-transformed
            for i in range(4,len(p0)):
                p0[i] = np.log(p0[i])
            utils.sanitize_initial_guess(p0, bounds)
            return p0
        else:
            if drug1_model is None:
                mask = np.where(d2==min(d2))
                drug1_model = hill.Hill.create_fit(d1[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)
            if drug2_model is None:
                mask = np.where(d1==min(d1))
                drug2_model = hill.Hill.create_fit(d2[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)
            
            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            
            # Get initial guess of E3 at E(d1_max, d2_max), if that point exists
            E3 = E[(d1==max(d1)) & (d2==max(d2))]
            if len(E3)>0: E3 = np.mean(E3)
            # TODO: Otherwise, guess E3 is the minimum value observed - THIS assumes the orientation of E is decreasing as d increases
            # Could infer orientation from sign of Y=(E0-E1)+(E0-E2). Y>0 means drug causes E to decrease. Y<0 means drug causes E to increase. Or let user set it
            else: E3 = np.min(E)
            
            p0 = [(E0_1+E0_2)/2., E1, E2, E3, np.log(h1), np.log(h2), np.log(C1), np.log(C2), 0, 0] #, 0, 0] # For gamma
            
            utils.sanitize_initial_guess(p0, bounds)

            return p0

    def E(self, d1, d2):
        if not self._is_parameterized():
            return 0
            #raise ModelNotParameterizedError()
        return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1, self.r2, self.alpha12, self.alpha21)

    def get_parameters(self):
        #return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, self.gamma12, self.gamma21, self.r1, self.r2
        return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, self.r1, self.r2


    def _C_to_r1r(self, C, h, r1):
        return r1*C**h

    def _r_to_C(self, h, r1r):
        return (r1r/r1)**(1./h)

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, r1, r2, alpha12, alpha21):

        d1h1 = d1**h1
        d2h2 = d2**h2
        alpha21d1h1 = (alpha21*d1)**h1
        alpha12d2h2 = (alpha12*d2)**h2
        r1r = r1*C1**h1
        r2r = r2*C2**h2

        U = r1r*r2r*(r1*(alpha21d1h1) + r1r + r2*(alpha12d2h2) + r2r)/(d1h1*r1**2*r2*(alpha12d2h2)*(alpha21d1h1) + d1h1*r1**2*r2r*(alpha21d1h1) + d1h1*r1*r1r*r2*(alpha12d2h2) + d1h1*r1*r1r*r2r + d1h1*r1*r2*r2r*(alpha12d2h2) + d1h1*r1*r2r**2 + d2h2*r1*r1r*r2*(alpha21d1h1) + d2h2*r1*r2**2*(alpha12d2h2)*(alpha21d1h1) + d2h2*r1*r2*r2r*(alpha21d1h1) + d2h2*r1r**2*r2 + d2h2*r1r*r2**2*(alpha12d2h2) + d2h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21d1h1) + r1r**2*r2r + r1r*r2*r2r*(alpha12d2h2) + r1r*r2r**2)

        A1 = r1*r2r*(d1h1*r1*(alpha21d1h1) + d1h1*r1r + d1h1*r2r + d2h2*r2*(alpha21d1h1))/(d1h1*r1**2*r2*(alpha12d2h2)*(alpha21d1h1) + d1h1*r1**2*r2r*(alpha21d1h1) + d1h1*r1*r1r*r2*(alpha12d2h2) + d1h1*r1*r1r*r2r + d1h1*r1*r2*r2r*(alpha12d2h2) + d1h1*r1*r2r**2 + d2h2*r1*r1r*r2*(alpha21d1h1) + d2h2*r1*r2**2*(alpha12d2h2)*(alpha21d1h1) + d2h2*r1*r2*r2r*(alpha21d1h1) + d2h2*r1r**2*r2 + d2h2*r1r*r2**2*(alpha12d2h2) + d2h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21d1h1) + r1r**2*r2r + r1r*r2*r2r*(alpha12d2h2) + r1r*r2r**2)

        A2 = r1r*r2*(d1h1*r1*(alpha12d2h2) + d2h2*r1r + d2h2*r2*(alpha12d2h2) + d2h2*r2r)/(d1h1*r1**2*r2*(alpha12d2h2)*(alpha21d1h1) + d1h1*r1**2*r2r*(alpha21d1h1) + d1h1*r1*r1r*r2*(alpha12d2h2) + d1h1*r1*r1r*r2r + d1h1*r1*r2*r2r*(alpha12d2h2) + d1h1*r1*r2r**2 + d2h2*r1*r1r*r2*(alpha21d1h1) + d2h2*r1*r2**2*(alpha12d2h2)*(alpha21d1h1) + d2h2*r1*r2*r2r*(alpha21d1h1) + d2h2*r1r**2*r2 + d2h2*r1r*r2**2*(alpha12d2h2) + d2h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21d1h1) + r1r**2*r2r + r1r*r2*r2r*(alpha12d2h2) + r1r*r2r**2)

        #A12 = r1*r2*(d1h1*r1*(alpha12d2h2)*(alpha21d1h1) + d1h1*r2r*(alpha12d2h2) + d2h2*r1r*(alpha21d1h1) + d2h2*r2*(alpha12d2h2)*(alpha21d1h1))/(d1h1*r1**2*r2*(alpha12d2h2)*(alpha21d1h1) + d1h1*r1**2*r2r*(alpha21d1h1) + d1h1*r1*r1r*r2*(alpha12d2h2) + d1h1*r1*r1r*r2r + d1h1*r1*r2*r2r*(alpha12d2h2) + d1h1*r1*r2r**2 + d2h2*r1*r1r*r2*(alpha21d1h1) + d2h2*r1*r2**2*(alpha12d2h2)*(alpha21d1h1) + d2h2*r1*r2*r2r*(alpha21d1h1) + d2h2*r1r**2*r2 + d2h2*r1r*r2**2*(alpha12d2h2) + d2h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21d1h1) + r1r**2*r2r + r1r*r2*r2r*(alpha12d2h2) + r1r*r2r**2)

        #A12 = 1-(U+A1+A2)
        
        return U*E0 + A1*E1 + A2*E2 + (1-(U+A1+A2))*E3

    def create_fit(d1, d2, E, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), E3_bounds=(-np.inf,np.inf), \
            alpha12_bounds=(0,np.inf), alpha21_bounds=(0,np.inf),   \
            gamma12_bounds=(0,np.inf), gamma21_bounds=(0,np.inf),   \
            r1=1., r2=1., **kwargs):

        model = MuSyC(E0_bounds=E0_bounds, E1_bounds=E1_bounds, E2_bounds=E2_bounds, E3_bounds=E3_bounds, h1_bounds=h1_bounds, h2_bounds=h2_bounds, C1_bounds=C1_bounds, C2_bounds=C2_bounds, alpha12_bounds=alpha12_bounds, alpha21_bounds=alpha21_bounds, gamma12_bounds=gamma12_bounds, gamma21_bounds=gamma21_bounds, r1=r1, r2=r2)
        model.fit(d, E, **kwargs)
        return model

    def __repr__(self):
        if not self._is_parameterized(): return "MuSyC()"
        return "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, r1=%0.2f, r2=%0.2f, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f, converged=%r)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1, self.r2, self.alpha12, self.alpha21, self.beta, self.converged)
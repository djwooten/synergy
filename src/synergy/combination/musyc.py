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

from .jacobians.musyc_jacobian import jacobian
from .parametric_base import ParametricModel

from .. import utils
from ..single import Hill

class MuSyC(ParametricModel):
    """Multidimensional Synergy of Combinations (MuSyC) is a drug synergy framework based on the law of mass action (doi: 10.1016/j.cels.2019.01.003, doi: 10.1101/683433). In MuSyC, synergy is parametrically defined as shifts in potency, efficacy, or cooperativity.

    alpha21 : float
        Synergistic potency ([0,1) = antagonism, (1,inf) = synergism).        At large concentrations of drug 2, the "effective dose" of drug 1 = alpha21*d1.
    
    alpha12 : float
        Synergistic potency ([0,1) = antagonism, (1,inf) = synergism).         At large concentrations of drug 1, the "effective dose" of drug 2 = alpha12*d2.

    beta : float
        Synergistic efficacy ((-inf,0) = antagonism, (0,inf) = synergism). At large concentrations of both drugs, the combination achieves an effect beta-% stronger (or weaker) than the stronger single-drug.

    gamma21 : float
        Synergistic cooperativity ([0,1) = antagonism, (1,inf) = synergism). At large concentrations of drug 2, the Hill slope of drug 1 = gamma21*h1

    gamma12 : float
        Synergistic cooperativity ([0,1) = antagonism, (1,inf) = synergism). At large concentrations of drug 1, the Hill slope of drug 2 = gamma12*h2

    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), E3_bounds=(-np.inf,np.inf), \
            alpha12_bounds=(0,np.inf), alpha21_bounds=(0,np.inf),   \
            gamma12_bounds=(0,np.inf), gamma21_bounds=(0,np.inf),   \
            r1=1., r2=1., E0=None, E1=None, E2=None, E3=None,   \
            h1=None, h2=None, C1=None, C2=None, oalpha12=None,       \
            oalpha21=None, gamma12=None, gamma21=None):
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
        #self.alpha12 = alpha12
        #self.alpha21 = alpha21
        self.alpha12 = MuSyC._prime_to_alpha(oalpha12, C2, gamma12)
        self.alpha21 = MuSyC._prime_to_alpha(oalpha21, C1, gamma21)
        #self.oalpha12 = MuSyC._alpha_to_prime(alpha12, C2, gamma12)
        #self.oalpha21 = MuSyC._alpha_to_prime(alpha21, C1, gamma21)
        self.gamma12 = gamma12
        self.gamma21 = gamma21
        if not None in [E1, E2, E3]:
            self.beta = (min(E1,E2)-E3) / (E0 - min(E1,E2))
        else:
            self.beta = None

        with np.errstate(divide='ignore'):
            self.logh1_bounds = (np.log(h1_bounds[0]), np.log(h1_bounds[1]))
            self.logC1_bounds = (np.log(C1_bounds[0]), np.log(C1_bounds[1]))
            self.logh2_bounds = (np.log(h2_bounds[0]), np.log(h2_bounds[1]))
            self.logC2_bounds = (np.log(C2_bounds[0]), np.log(C2_bounds[1]))
            
            self.logalpha12_bounds = (np.log(alpha12_bounds[0]), np.log(alpha12_bounds[1]))
            self.logalpha21_bounds = (np.log(alpha21_bounds[0]), np.log(alpha21_bounds[1]))

            self.loggamma12_bounds = (np.log(gamma12_bounds[0]), np.log(gamma12_bounds[1]))
            self.loggamma21_bounds = (np.log(gamma21_bounds[0]), np.log(gamma21_bounds[1]))

        self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), self.r1, self.r2, np.exp(logalpha12), np.exp(logalpha21), np.exp(loggamma12), np.exp(loggamma21))

        self.jacobian_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21: jacobian(d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1, self.r2, logalpha12, logalpha21, loggamma12, loggamma21)

        self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logalpha12_bounds, self.logalpha21_bounds, self.loggamma12_bounds, self.loggamma21_bounds))

    def _alpha_to_prime(alpha, C, gamma):
        if None in [alpha, C, gamma]: return None
        #return alpha*C**((gamma-1)/gamma)
        return alpha*np.power(C,(gamma-1)/gamma)
    
    def _prime_to_alpha(prime, C, gamma):
        if None in [prime, C, gamma]: return None
        #return prime*C**((1-gamma)/gamma)
        return prime*np.power(C,(1-gamma)/gamma)

    def _get_initial_guess(self, d1, d2, E, drug1_model=None, drug2_model=None, p0=None):
        
        if p0 is None:
            if drug1_model is None:
                mask = np.where(d2==min(d2))
                drug1_model = Hill.create_fit(d1[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)
            if drug2_model is None:
                mask = np.where(d1==min(d1))
                drug2_model = Hill.create_fit(d2[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)
            
            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            
            #TODO: E orientation
            # Get initial guess of E3 at E(d1_max, d2_max), if that point exists
            E3 = E[(d1==max(d1)) & (d2==max(d2))]
            if len(E3)>0: E3 = np.mean(E3)

            # Otherwise guess E3 is the minimum E observed
            else: E3 = np.min(E)
            
            p0 = [(E0_1+E0_2)/2., E1, E2, E3, h1, h2, C1, C2, 1, 1, 1, 1]
            
        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        
        E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21 = params
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        alpha12 = np.exp(logalpha12)
        alpha21 = np.exp(logalpha21)
        gamma12 = np.exp(loggamma12)
        gamma21 = np.exp(loggamma21)

        oalpha12 = MuSyC._alpha_to_prime(alpha12, C2, gamma12)
        oalpha21 = MuSyC._alpha_to_prime(alpha21, C1, gamma21)

        return E0, E1, E2, E3, h1, h2, C1, C2, oalpha12, oalpha21, gamma12, gamma21

    def _transform_params_to_fit(self, params):
        
        E0, E1, E2, E3, h1, h2, C1, C2, oalpha12, oalpha21, gamma12, gamma21 = params

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)
        alpha12 = MuSyC._prime_to_alpha(oalpha12, C2, gamma12)
        alpha21 = MuSyC._prime_to_alpha(oalpha21, C1, gamma21)
        logalpha12 = np.log(alpha12)
        logalpha21 = np.log(alpha21)
        loggamma12 = np.log(gamma12)
        loggamma21 = np.log(gamma21)

        return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21

    def E(self, d1, d2):
        if not self._is_parameterized():
            return 0
            #raise ModelNotParameterizedError()
        return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1, self.r2, self.alpha12, self.alpha21, self.gamma12, self.gamma21)

    def get_parameters(self):
        oalpha12 = MuSyC._alpha_to_prime(self.alpha12, self.C2, self.gamma12)
        oalpha21 = MuSyC._alpha_to_prime(self.alpha21, self.C1, self.gamma21)

        return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, oalpha12, oalpha21, self.gamma12, self.gamma21
    
    def _set_parameters(self, popt):
        self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, oalpha12, oalpha21, self.gamma12, self.gamma21 = popt

        self.alpha12 = MuSyC._prime_to_alpha(oalpha12, self.C2, self.gamma12)
        self.alpha21 = MuSyC._prime_to_alpha(oalpha21, self.C1, self.gamma21)

        

    def _C_to_r1r(self, C, h, r1):
        return r1*C**h

    def _r_to_C(self, h, r1r):
        return (r1r/r1)**(1./h)

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, r1, r2, alpha12, alpha21, gamma12, gamma21):

        #d1h1 = d1**h1
        #d2h2 = d2**h2
        d1h1 = np.power(d1,h1)
        d2h2 = np.power(d2,h2)

        #C1h1 = C1**h1
        #C2h2 = C2**h2
        C1h1 = np.power(C1,h1)
        C2h2 = np.power(C2,h2)

        #alpha21d1gamma21h1 = (alpha21*d1)**(gamma21*h1)
        #alpha12d2gamma12h2 = (alpha12*d2)**(gamma12*h2)
        alpha21d1gamma21h1 = np.power(alpha21*d1, gamma21*h1)
        alpha12d2gamma12h2 = np.power(alpha12*d2, gamma12*h2)

        #C12h1 = C1**(2*h1)
        #C22h2 = C2**(2*h2)
        C12h1 = np.power(C1,2*h1)
        C22h2 = np.power(C2,2*h2)

        # ********** U ********

        U = r1*r2*(r1*alpha21d1gamma21h1 + r1*C1h1 + r2*alpha12d2gamma12h2 + r2*C2h2)*C1h1*C2h2/(d1h1*r1**2*r2*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**2*r2*alpha21d1gamma21h1*C2h2 + d1h1*r1**2*r2*alpha12d2gamma12h2*C1h1 + d1h1*r1**2*r2*C1h1*C2h2 + d1h1*r1*r2**2*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**2*C22h2 + d2h2*r1**2*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**2*r2*C12h1 + d2h2*r1*r2**2*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r1*r2**2*alpha21d1gamma21h1*C2h2 + d2h2*r1*r2**2*alpha12d2gamma12h2*C1h1 + d2h2*r1*r2**2*C1h1*C2h2 + r1**2*r2*alpha21d1gamma21h1*C1h1*C2h2 + r1**2*r2*C12h1*C2h2 + r1*r2**2*alpha12d2gamma12h2*C1h1*C2h2 + r1*r2**2*C1h1*C22h2)

        # ********** A1 ********

        A1 = r1*r2*(d1h1*r1*alpha21d1gamma21h1 + d1h1*r1*C1h1 + d1h1*r2*C2h2 + d2h2*r2*alpha21d1gamma21h1)*C2h2/(d1h1*r1**2*r2*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**2*r2*alpha21d1gamma21h1*C2h2 + d1h1*r1**2*r2*alpha12d2gamma12h2*C1h1 + d1h1*r1**2*r2*C1h1*C2h2 + d1h1*r1*r2**2*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**2*C22h2 + d2h2*r1**2*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**2*r2*C12h1 + d2h2*r1*r2**2*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r1*r2**2*alpha21d1gamma21h1*C2h2 + d2h2*r1*r2**2*alpha12d2gamma12h2*C1h1 + d2h2*r1*r2**2*C1h1*C2h2 + r1**2*r2*alpha21d1gamma21h1*C1h1*C2h2 + r1**2*r2*C12h1*C2h2 + r1*r2**2*alpha12d2gamma12h2*C1h1*C2h2 + r1*r2**2*C1h1*C22h2)

        # ********** A2 ********

        A2 = r1*r2*(d1h1*r1*alpha12d2gamma12h2 + d2h2*r1*C1h1 + d2h2*r2*alpha12d2gamma12h2 + d2h2*r2*C2h2)*C1h1/(d1h1*r1**2*r2*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**2*r2*alpha21d1gamma21h1*C2h2 + d1h1*r1**2*r2*alpha12d2gamma12h2*C1h1 + d1h1*r1**2*r2*C1h1*C2h2 + d1h1*r1*r2**2*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**2*C22h2 + d2h2*r1**2*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**2*r2*C12h1 + d2h2*r1*r2**2*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r1*r2**2*alpha21d1gamma21h1*C2h2 + d2h2*r1*r2**2*alpha12d2gamma12h2*C1h1 + d2h2*r1*r2**2*C1h1*C2h2 + r1**2*r2*alpha21d1gamma21h1*C1h1*C2h2 + r1**2*r2*C12h1*C2h2 + r1*r2**2*alpha12d2gamma12h2*C1h1*C2h2 + r1*r2**2*C1h1*C22h2)
        
        return U*E0 + A1*E1 + A2*E2 + (1-(U+A1+A2))*E3
    
    def create_fit(d1, d2, E, h_bounds=(1e-3,1e3), C_bounds=(0,np.inf),     \
            E_bounds=(-np.inf,np.inf), oalpha_bounds=(1e-5,1e5),            \
            gamma_bounds=(1e-5,1e5), **kwargs):

        dmin = min(min(d1), min(d2))
        dmax = max(max(d1), max(d2))
        alpha_lb = MuSyC._prime_to_alpha(oalpha_bounds[0], dmin, gamma_bounds[0])
        alpha_ub = MuSyC._prime_to_alpha(oalpha_bounds[1], dmax, gamma_bounds[1])
        alpha_bounds = (alpha_lb, alpha_ub)

        model = MuSyC(E0_bounds=E_bounds, E1_bounds=E_bounds, E2_bounds=E_bounds, E3_bounds=E_bounds, h1_bounds=h_bounds, h2_bounds=h_bounds, C1_bounds=C_bounds, C2_bounds=C_bounds, alpha12_bounds=alpha_bounds, alpha21_bounds=alpha_bounds, gamma12_bounds=gamma_bounds, gamma21_bounds=gamma_bounds)

        model.fit(d1, d2, E, **kwargs)
        return model

    def get_parameter_range(self, confidence_interval=95):
        if not self._is_parameterized():
            return None
        if not self.converged:
            return None
        if confidence_interval < 0 or confidence_interval > 100:
            return None
        if self.bootstrap_parameters is None:
            return None

        lb = (100-confidence_interval)/2.
        ub = 100-lb
        bp = np.array(self.bootstrap_parameters, copy=True)
        E0 = self.bootstrap_parameters[:,0]
        E1 = self.bootstrap_parameters[:,1]
        E2 = self.bootstrap_parameters[:,2]
        E3 = self.bootstrap_parameters[:,3]
        
        beta = (np.minimum(E1,E2)-E3) / (E0 - np.minimum(E1,E2))
        bp = np.insert(self.bootstrap_parameters, 10, values=beta, axis=1)
        return np.percentile(bp, [lb, ub], axis=0)

    def __repr__(self):
        if not self._is_parameterized(): return "MuSyC()"
        
        oalpha12 = MuSyC._alpha_to_prime(self.alpha12, self.C2, self.gamma12)
        oalpha21 = MuSyC._alpha_to_prime(self.alpha21, self.C1, self.gamma21)

        beta = (min(self.E1,self.E2)-self.E3) / (self.E0 - min(self.E1,self.E2))

        return "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, oalpha12=%0.2f, oalpha21=%0.2f, beta=%0.2f, gamma12=%0.2f, gamma21=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, oalpha12, oalpha21, beta, self.gamma12, self.gamma21)
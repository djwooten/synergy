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
            r1r=1., r2r=1., E0=None, E1=None, E2=None, E3=None,     \
            h1=None, h2=None, C1=None, C2=None, alpha12=None,       \
            alpha21=None, gamma12=None, gamma21=None, variant="full"):
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

        self.variant = variant

        self.r1r = r1r
        self.r2r = r2r
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


        if variant == "full":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), self.r1r, self.r2r, np.exp(logalpha12), np.exp(logalpha21), np.exp(loggamma12), np.exp(loggamma21))

            self.jacobian_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21: jacobian(d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1r, self.r2r, logalpha12, logalpha21, loggamma12, loggamma21)

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logalpha12_bounds, self.logalpha21_bounds, self.loggamma12_bounds, self.loggamma21_bounds))
        
        elif variant == "no_gamma":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), self.r1r, self.r2r, np.exp(logalpha12), np.exp(logalpha21), 1, 1)

            self.jacobian_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21: jacobian(d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1r, self.r2r, logalpha12, logalpha21, 0, 0)[:,:-2]

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logalpha12_bounds, self.logalpha21_bounds))


    def _get_initial_guess(self, d1, d2, E, drug1_model, drug2_model, p0=None):
        

        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:
            # Sanitize single-drug models
            default_class, expected_superclass = self._get_single_drug_classes()

            drug1_model = utils.sanitize_single_drug_model(drug1_model, default_class, expected_superclass=expected_superclass, E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)

            drug2_model = utils.sanitize_single_drug_model(drug2_model, default_class, expected_superclass=expected_superclass, E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)

            # Fit the single drug models if they were not pre-fit by the user
            if not drug1_model.is_fit():
                mask = np.where(d2==min(d2))
                drug1_model.fit(d1[mask], E[mask])
            if not drug2_model.is_fit():
                mask = np.where(d1==min(d1))
                drug2_model.fit(d2[mask], E[mask])

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
        
            if self.variant == "no_gamma":
                p0 = p0[:-2]

        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        
        if self.variant == "no_gamma":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21 = params
        else:
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21 = params
            gamma12 = np.exp(loggamma12)
            gamma21 = np.exp(loggamma21)
        
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        alpha12 = np.exp(logalpha12)
        alpha21 = np.exp(logalpha21)
        
        if self.variant == "no_gamma":
            return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21
        
        return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21

    def _transform_params_to_fit(self, params):
        
        if self.variant == "no_gamma":
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21 = params
        else:
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21 = params
            loggamma12 = np.log(gamma12)
            loggamma21 = np.log(gamma21)

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)
        logalpha12 = np.log(alpha12)
        logalpha21 = np.log(alpha21)

        if self.variant == "no_gamma":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21

        return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21

    def E(self, d1, d2):
        if not self._is_parameterized():
            return None

        if self.variant == "no_gamma":
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1r, self.r2r, self.alpha12, self.alpha21, 1, 1)
        
        else:
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1r, self.r2r, self.alpha12, self.alpha21, self.gamma12, self.gamma21)

    def _get_parameters(self):
        if self.variant == "no_gamma":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21
        else:
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, self.gamma12, self.gamma21
    
    def _set_parameters(self, popt):
        if self.variant == "no_gamma":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21 = popt
        else:
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, self.gamma12, self.gamma21 = popt

    def _get_single_drug_classes(self):
        return Hill, Hill

    def _C_to_r1(self, C, h, r1r):
        return r1r/np.power(C,h)

    @DeprecationWarning
    def _C_to_r1r(self, C, h, r1):
        return r1*C**h

    @DeprecationWarning
    def _r_to_C(self, h, r1r):
        return (r1r/r1)**(1./h)

    def _reference_E(self, d1, d2):
        if not self._is_parameterized():
            return None
        return self._model(d1, d2, self.E0, self.E1, self.E2, min(self.E1,self.E2), self.h1, self.h2, self.C1, self.C2, self.r1r, self.r2r, 1, 1, 1, 1)

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, r1r, r2r, alpha12, alpha21, gamma12, gamma21):

        d1h1 = np.power(d1,h1)
        d2h2 = np.power(d2,h2)

        C1h1 = np.power(C1,h1)
        C2h2 = np.power(C2,h2)

        r1 = r1r/C1h1
        r2 = r2r/C2h2

        alpha21d1gamma21h1 = np.power(alpha21*d1, gamma21*h1)
        alpha12d2gamma12h2 = np.power(alpha12*d2, gamma12*h2)

        C12h1 = np.power(C1,2*h1)
        C22h2 = np.power(C2,2*h2)

        # ********** U ********

        U=(r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)/(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*r1*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*C2h2+d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d1h1*np.power(r1,(gamma21+1))*np.power(r2,gamma12)*alpha21d1gamma21h1*alpha12d2gamma12h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r1,(gamma21+1))*r2*alpha21d1gamma21h1*C1h1+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*np.power(r2,(gamma12+1))*alpha21d1gamma21h1*alpha12d2gamma12h2+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)

        #**********E1********

        A1=(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12))/(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*r1*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*C2h2+d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d1h1*np.power(r1,(gamma21+1))*np.power(r2,gamma12)*alpha21d1gamma21h1*alpha12d2gamma12h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r1,(gamma21+1))*r2*alpha21d1gamma21h1*C1h1+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*np.power(r2,(gamma12+1))*alpha21d1gamma21h1*alpha12d2gamma12h2+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)

        #**********E2********

        A2=(d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21))/(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*r1*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*C2h2+d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d1h1*np.power(r1,(gamma21+1))*np.power(r2,gamma12)*alpha21d1gamma21h1*alpha12d2gamma12h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r1,(gamma21+1))*r2*alpha21d1gamma21h1*C1h1+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*np.power(r2,(gamma12+1))*alpha21d1gamma21h1*alpha12d2gamma12h2+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)

        
        return U*E0 + A1*E1 + A2*E2 + (1-(U+A1+A2))*E3
    
    @staticmethod
    def _get_beta(E0, E1, E2, E3):
        """Calculates synergistic efficacy, a synergy parameter derived from E parameters.
        """
        strongest_E = np.amin(np.asarray([E1,E2]), axis=0)
        beta = (strongest_E-E3) / (E0 - strongest_E)
        return beta


    def get_parameters(self, confidence_interval=95):
        if not self._is_parameterized():
            return None
        
        #beta = (min(self.E1,self.E2)-self.E3) / (self.E0 - min(self.E1,self.E2))
        beta = MuSyC._get_beta(self.E0, self.E1, self.E2, self.E3)

        if self.converged and self.bootstrap_parameters is not None:
            parameter_ranges = self.get_parameter_range(confidence_interval=confidence_interval)
        else:
            parameter_ranges = None

        params = dict()
        params['E0'] = [self.E0, ]
        params['E1'] = [self.E1, ]
        params['E2'] = [self.E2, ]
        params['E3'] = [self.E3, ]
        params['h1'] = [self.h1, ]
        params['h2'] = [self.h2, ]
        params['C1'] = [self.C1, ]
        params['C2'] = [self.C2, ]
        params['beta'] = [beta, ]
        params['alpha12'] = [self.alpha12, ]
        params['alpha21'] = [self.alpha21, ]
        if self.variant != "no_gamma":
            params['gamma12'] = [self.gamma12, ]
            params['gamma21'] = [self.gamma21, ]

        if parameter_ranges is not None:
            params['E0'].append(parameter_ranges[:,0])
            params['E1'].append(parameter_ranges[:,1])
            params['E2'].append(parameter_ranges[:,2])
            params['E3'].append(parameter_ranges[:,3])
            params['h1'].append(parameter_ranges[:,4])
            params['h2'].append(parameter_ranges[:,5])
            params['C1'].append(parameter_ranges[:,6])
            params['C2'].append(parameter_ranges[:,7])
            params['alpha12'].append(parameter_ranges[:,8])
            params['alpha21'].append(parameter_ranges[:,9])
            if self.variant != "no_gamma":
                params['gamma12'].append(parameter_ranges[:,10])
                params['gamma21'].append(parameter_ranges[:,11])

            bsE0 = self.bootstrap_parameters[:,0]
            bsE1 = self.bootstrap_parameters[:,1]
            bsE2 = self.bootstrap_parameters[:,2]
            bsE3 = self.bootstrap_parameters[:,3]
            beta_bootstrap = MuSyC._get_beta(bsE0, bsE1, bsE2, bsE3)

            beta_bootstrap = np.percentile(beta_bootstrap, [(100-confidence_interval)/2, 50+confidence_interval/2])
            params['beta'].append(beta_bootstrap)    
        return params
    
    def summary(self, confidence_interval=95, tol=0.01):
        pars = self.get_parameters(confidence_interval=confidence_interval)
        if pars is None:
            return None
        
        ret = []
        keys = pars.keys()
        # beta
        for key in keys:
            if "beta" in key:
                l = pars[key]
                if len(l)==1:
                    if l[0] < -tol:
                        ret.append("%s\t%0.2f\t(<0) antagonistic"%(key, l[0]))
                    elif l[0] > tol:
                        ret.append("%s\t%0.2f\t(>0) synergistic"%(key, l[0]))
                else:
                    v = l[0]
                    lb,ub = l[1]
                    if v < -tol and lb < -tol and ub < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<0) antagonistic"%(key, v,lb,ub))
                    elif v > tol and lb > tol and ub > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>0) synergistic"%(key, v,lb,ub))
        # alpha
        for key in keys:
            if "alpha" in key:
                l = pars[key]
                if len(l)==1:
                    if np.log10(l[0]) < -tol:
                        ret.append("%s\t%0.2f\t(<1) antagonistic"%(key, l[0]))
                    elif np.log10(l[0]) > tol:
                        ret.append("%s\t%0.2f\t(>1) synergistic"%(key, l[0]))
                else:
                    v = l[0]
                    lb,ub = l[1]
                    if np.log10(v) < -tol and np.log10(lb) < -tol and np.log10(ub) < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<1) antagonistic"%(key, v,lb,ub))
                    elif np.log10(v) > tol and np.log10(lb) > tol and np.log10(ub) > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>1) synergistic"%(key, v,lb,ub))

        # gamma
        for key in keys:
            if "gamma" in key:
                l = pars[key]
                if len(l)==1:
                    if np.log10(l[0]) < -tol:
                        ret.append("%s\t%0.2f\t(<1) antagonistic"%(key, l[0]))
                    elif np.log10(l[0]) > tol:
                        ret.append("%s\t%0.2f\t(>1) synergistic"%(key, l[0]))
                else:
                    v = l[0]
                    lb,ub = l[1]
                    if np.log10(v) < -tol and np.log10(lb) < -tol and np.log10(ub) < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<1) antagonistic"%(key, v,lb,ub))
                    elif np.log10(v) > tol and np.log10(lb) > tol and np.log10(ub) > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>1) synergistic"%(key, v,lb,ub))
        if len(ret)>0:
            return "\n".join(ret)
        else:
            return "No synergy or antagonism detected with %d percent confidence interval"%(int(confidence_interval))

    def __repr__(self):
        if not self._is_parameterized(): return "MuSyC()"
        
        #beta = (min(self.E1,self.E2)-self.E3) / (self.E0 - min(self.E1,self.E2))
        beta = MuSyC._get_beta(self.E0, self.E1, self.E2, self.E3)

        if self.variant == "no_gamma":
            return "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, beta)
        return "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f, gamma12=%0.2f, gamma21=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, beta, self.gamma12, self.gamma21)
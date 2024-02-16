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
from ..single import Hill_2P
from .parametric_base import ParametricModel

class Zimmer(ParametricModel):
    """The Effective Dose Model from Zimmer et al (doi: 10.1073/pnas.1606301113). This model uses the multiplicative survival principle (i.e., Bliss), but adds a parameter for each drug describing how it affects the potency of the other. Specifically, given doses d1 and d2, this model translates them to "effective" doses using the following system of equations

                            d1
    d1_eff =  --------------------------------
              1 + a12*(1/(1+(d2_eff/C2)^(-1)))

                            d2
    d2_eff =  --------------------------------
              1 + a21*(1/(1+(d1_eff/C1)^(-1)))

    Synergy by Zimmer is described by these parameters

    a12 : float
        (-(1+(d2_eff/C2))/(d2_eff/C2),0)=synergism, (0,inf)=antagonism.         Describes how drug 2 affects the effective dose of drug 1.

    a21 : float
        (-(1+(d1_eff/C1))/(d1_eff/C1),0)=synergism, (0,inf)=antagonism. Describes how drug 1 affects the effective dose of drug 2.
        
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

        self.fit_function = lambda d, logh1, logh2, logC1, logC2, a12, a21: self._model(d[0], d[1], np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), a12, a21)

        self.bounds = tuple(zip(self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.a12_bounds, self.a21_bounds))


    def _get_initial_guess(self, d1, d2, E, drug1_model, drug2_model, p0=None):
        
        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:
            # Sanitize single-drug models
            default_class, expected_superclass = self._get_single_drug_classes()

            drug1_model = utils.sanitize_single_drug_model(drug1_model, default_class, expected_superclass=expected_superclass, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)

            drug2_model = utils.sanitize_single_drug_model(drug2_model, default_class, expected_superclass=expected_superclass, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)

            # Fit the single drug models if they were not pre-fit by the user
            if not drug1_model.is_fit():
                mask = np.where(d2==min(d2))
                drug1_model.fit(d1[mask], E[mask])
            if not drug2_model.is_fit():
                mask = np.where(d1==min(d1))
                drug2_model.fit(d2[mask], E[mask])
            
            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            h1, C1 = drug1_model.get_parameters()
            h2, C2 = drug2_model.get_parameters()
                        
            p0 = [h1, h2, C1, C2, 0, 0]
            
        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        logh1, logh2, logC1, logC2, a12, a21 = params
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        return h1, h2, C1, C2, a12, a21

    def _transform_params_to_fit(self, params):
        h1, h2, C1, C2, a12, a21 = params

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)
        
        return logh1, logh2, logC1, logC2, a12, a21
        

    def _set_parameters(self, popt):
        self.h1, self.h2, self.C1, self.C2, self.a12, self.a21 = popt

    def E(self, d1, d2):
        if not self._is_parameterized():
            # ERROR
            return 0
        return self._model(d1, d2, self.h1, self.h2, self.C1, self.C2, self.a12, self.a21)

    def _reference_E(self, d1, d2):
        if not self._is_parameterized():
            return None
        return self._model(d1, d2, self.h1, self.h2, self.C1, self.C2, 0, 0)

    def _model(self, d1, d2, h1, h2, C1, C2, a12, a21):
        A = d2 + C2*(a21+1) + d2*a12
        B = d2*C1 + C1*C2 + a12*d2*C1 - d1*(d2+C2*(a21+1))
        C = -d1*(d2*C1 + C1*C2)

        d1p = (-B + np.sqrt(np.power(B,2.) - 4*A*C)) / (2.*A)
        d2p = d2 / (1. + a21 / (1. + C1/d1p))
        
        return (1 - np.power(d1p,h1)/(np.power(C1,h1)+np.power(d1p,h1))) * (1 - np.power(d2p,h2)/(np.power(C2,h2)+np.power(d2p,h2)))

    def _get_parameters(self):
        return self.h1, self.h2, self.C1, self.C2, self.a12, self.a21

    def _get_single_drug_classes(self):
        return Hill_2P, Hill_2P

    def get_parameters(self, confidence_interval=95):
        if not self._is_parameterized():
            return None
        
        if self.converged and self.bootstrap_parameters is not None:
            parameter_ranges = self.get_parameter_range(confidence_interval=confidence_interval)
        else:
            parameter_ranges = None

        params = dict()
        params['h1'] = [self.h1, ]
        params['h2'] = [self.h2, ]
        params['C1'] = [self.C1, ]
        params['C2'] = [self.C2, ]
        params['a12'] = [self.a12, ]
        params['a21'] = [self.a21, ]
        
        if parameter_ranges is not None:
            params['h1'].append(parameter_ranges[:,0])
            params['h2'].append(parameter_ranges[:,1])
            params['C1'].append(parameter_ranges[:,2])
            params['C2'].append(parameter_ranges[:,3])
            params['a12'].append(parameter_ranges[:,4])
            params['a21'].append(parameter_ranges[:,5])

        return params
    
    def summary(self, confidence_interval=95, tol=0.01):
        pars = self.get_parameters(confidence_interval=confidence_interval)
        if pars is None:
            return None
        
        ret = []
        for key in ['a12','a21']:
            l = pars[key]
            if len(l)==1:
                if l[0] < -tol:
                    ret.append("%s\t%0.2f\t(<0) synergistic"%(key, l[0]))
                elif l[0] > tol:
                    ret.append("%s\t%0.2f\t(>0) antagonistic"%(key, l[0]))
            else:
                v = l[0]
                lb,ub = l[1]
                if v < -tol and lb < -tol and ub < -tol:
                    ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<0) synergistic"%(key, v,lb,ub))
                elif v > tol and lb > tol and ub > tol:
                    ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>0) antagonistic"%(key, v,lb,ub))

        if len(ret)>0:
            return "\n".join(ret)
        else:
            return "No synergy or antagonism detected with %d percent confidence interval"%(int(confidence_interval))

    def __repr__(self):
        if not self._is_parameterized(): return "Zimmer()"
        return "Zimmer(h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, a12=%0.2f, a21=%0.2f)"%(self.h1, self.h2, self.C1, self.C2, self.a12, self.a21)
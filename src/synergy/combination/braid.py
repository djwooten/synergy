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
    
    kappa and delta are the BRAID synergy parameters, though E3 is related to how much more effective the combination is than either drug alone.

    Parameters
    ----------
    [X]_bounds : tuple
        Upper and lower bounds for each parameter to constrain fits.

    variant : str , default="kappa"
        Options "kappa", "delta", "both". BRAID has model versions that fit synergy using the parameter "kappa", the parameter "delta", or both. The standard version only fits kappa, but the other variants are available.

    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), E3_bounds=(-np.inf,np.inf), \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf), kappa_bounds=(-2, np.inf), delta_bounds=(0, np.inf), E0=None, E1=None, E2=None, E3=None, h1=None, h2=None, C1=None, C2=None, kappa=None, delta=None, variant="kappa"):

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

        self.variant = variant
        if variant == "kappa":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), kappa, 1)

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.kappa_bounds))
        
        elif variant == "delta":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logdelta: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), 0, np.exp(logdelta))

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logdelta_bounds))
        
        elif variant == "both":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), kappa, np.exp(logdelta))

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.kappa_bounds, self.logdelta_bounds))

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian = True, p0=None, bootstrap_iterations=0, **kwargs):
        super().fit(d1, d2, E, drug1_model=drug1_model, drug2_model=drug2_model, use_jacobian=use_jacobian, p0=p0, bootstrap_iterations=bootstrap_iterations, **kwargs)

        # The way BRAID fits E3 uses a max() function that can actually make E3 not matter at all, over a potentially large range. Untreated, this can cause its uncertainty to explode. But E3 should be defined as the value that gives the greatest maximum E_range. Here we must manually fix it for bootstrapping.
        if self.bootstrap_parameters is not None and self.bootstrap_parameters.shape[0]>0:
            for i in range(self.bootstrap_parameters.shape[0]):
                E0, E1, E2, E3, = self.bootstrap_parameters[i,:4]
                delta_Es = [E1-E0, E2-E0, E3-E0]
                max_delta_E_index = np.argmax(np.abs(delta_Es))
                max_delta_E = delta_Es[max_delta_E_index]
                E3 = max_delta_E + E0
                self.bootstrap_parameters[i,3] = E3


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
            E0 = (E0_1+E0_2)/2
            
            E3 = E[(d1==max(d1)) & (d2==max(d2))]
            if len(E3)>0: E3 = np.mean(E3)
            else: E3 = np.min(E)

            if self.variant == "kappa":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0]
            elif self.variant == "delta":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 1]
            elif self.variant == "both":
                p0 = [E0, E1, E2, E3, h1, h2, C1, C2, 0, 1]
            
        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        if self.variant == "kappa":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa = params
        elif self.variant == "delta":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logdelta = params
            delta = np.exp(logdelta)
        elif self.variant == "both":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta = params
            delta = np.exp(logdelta)
        else:
            return None
        
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)

        if self.variant == "kappa":
            return E0, E1, E2, E3, h1, h2, C1, C2, kappa
        elif self.variant == "delta":
            return E0, E1, E2, E3, h1, h2, C1, C2, delta
        elif self.variant == "both":
            return E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta
        

    def _transform_params_to_fit(self, params):
        if self.variant == "kappa":
            E0, E1, E2, E3, h1, h2, C1, C2, kappa  = params
        elif self.variant == "delta":
            E0, E1, E2, E3, h1, h2, C1, C2, delta  = params
            logdelta = np.log(delta)
        elif self.variant == "both":
            E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta  = params
            logdelta = np.log(delta)
        else: return None

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)

        if self.variant == "kappa":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa
        elif self.variant == "delta":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logdelta
        elif self.variant == "both":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, kappa, logdelta
        
    def _set_parameters(self, popt):
        if self.variant == "kappa":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa = popt
        elif self.variant == "delta":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta = popt
        elif self.variant == "both":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta = popt

        # E3 is, by definition, the value of E that gives the greatest delta_E. In fitting, the max() function can make E3 have no impact, leading to very sloppy output. Thus here we correct it by setting E3 to whichever E gives the greatest delta_E.

        delta_Es = [self.E1-self.E0, self.E2-self.E0, self.E3-self.E0]
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]
        self.E3 = max_delta_E + self.E0
        

    def E(self, d1, d2):
        if not self._is_parameterized():
            # ERROR
            return 0

        if self.variant == "kappa":
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, 1)
        elif self.variant == "delta":
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, 0, self.delta)
        elif self.variant == "both":
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta)
        
    def _reference_E(self, d1, d2):
        if not self._is_parameterized():
            return None
        return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, 0, 1)
    
    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, kappa, delta):
        """
From the braidrm R package (https://rdrr.io/cran/braidrm/man/evalBRAIDrsm.html)

The parameters of this equation must satisfy h1>0, h2>0, delta>0, kappa>-2, sign(E3-E0)=sign(E1-E0)=sign(E2-E0), |E3-E0|>=|E1-E0|, and |E3-E0|>=|E2-E0|.
        """
        delta_Es = [E1-E0, E2-E0, E3-E0]
        max_delta_E_index = np.argmax(np.abs(delta_Es))
        max_delta_E = delta_Es[max_delta_E_index]

        h = np.sqrt(h1*h2)
        power = 1/(delta*h)
        
        D1 = (E1-E0)/max_delta_E * np.power(d1/C1,h1)/(1+(1-(E1-E0)/max_delta_E)*np.power(d1/C1,h1))
        
        D2 = (E2-E0)/max_delta_E*np.power(d2/C2,h2)/(1+(1-(E2-E0)/max_delta_E)*np.power(d2/C2,h2))
        
        D = np.power(D1,power) + np.power(D2,power) +kappa*np.sqrt(np.power(D1,power)*np.power(D2,power))
        
        return E0 + max_delta_E/(1+np.power(D,-delta*h))

    def _get_parameters(self):
        if self.variant == "kappa":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa
        elif self.variant == "delta":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta
        elif self.variant == "both":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta

    def _get_single_drug_classes(self):
        return Hill, Hill

    def get_parameters(self, confidence_interval=95):
        if not self._is_parameterized():
            return None
        
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
        if self.variant in ["kappa","both"]:
            params['kappa'] = [self.kappa, ]
        if self.variant in ["delta","both"]:
            params['delta'] = [self.delta, ]
        
        if parameter_ranges is not None:
            params['E0'].append(parameter_ranges[:,0])
            params['E1'].append(parameter_ranges[:,1])
            params['E2'].append(parameter_ranges[:,2])
            params['E3'].append(parameter_ranges[:,3])
            params['h1'].append(parameter_ranges[:,4])
            params['h2'].append(parameter_ranges[:,5])
            params['C1'].append(parameter_ranges[:,6])
            params['C2'].append(parameter_ranges[:,7])
            if self.variant == "kappa":
                params['kappa'].append(parameter_ranges[:,8])
            elif self.variant == "delta":
                params['delta'].append(parameter_ranges[:,8])
            elif self.variant == "both":
                params['kappa'].append(parameter_ranges[:,8])
                params['delta'].append(parameter_ranges[:,9])

        return params
    
    def summary(self, confidence_interval=95, tol=0.01):
        pars = self.get_parameters(confidence_interval=confidence_interval)
        if pars is None:
            return None
        
        ret = []
        key = "kappa"
        if key in pars:
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

        key = "delta"
        if key in pars:
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
        if not self._is_parameterized(): return "BRAID()"

        if self.variant == "kappa":
            return "BRAID(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, kappa=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa)
        elif self.variant == "delta":
            return "BRAID(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, delta=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.delta)
        elif self.variant == "both":
            return "BRAID(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, kappa=%0.2f, delta=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.kappa, self.delta)
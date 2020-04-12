from scipy.optimize import curve_fit
import numpy as np
import multiprocessing
import synergy.utils.utils as utils

class Hill:
    """
    """
    def __init__(self, E0=None, Emax=None, h=None, C=None, E0_bounds=(-np.inf, np.inf), Emax_bounds=(-np.inf, np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf)):
        self.E0 = E0
        self.Emax = Emax
        self.h = h
        self.C = C
        
        self.E0_bounds=E0_bounds
        self.Emax_bounds=Emax_bounds
        self.h_bounds=h_bounds
        self.C_bounds=C_bounds
        with np.errstate(divide='ignore'):
            self.logh_bounds = (np.log(h_bounds[0]), np.log(h_bounds[1]))
            self.logC_bounds = (np.log(C_bounds[0]), np.log(C_bounds[1]))

        self.jacobian_queue = multiprocessing.Queue()
        self.converged = False
    
    def fit(self, d, E, use_jacobian=True, **kwargs):
        f = lambda d, E0, E1, logh, logC: self._model(d, E0, E1, np.exp(logh), np.exp(logC))

        bounds = tuple(zip(self.E0_bounds, self.Emax_bounds, self.logh_bounds, self.logC_bounds))

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
            p0[2] = np.log(p0[2])
            p0[3] = np.log(p0[3])
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0
        else:
            p0 = [max(E), min(E), 0, np.log(np.median(d))]
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0

        try:
            if use_jacobian:
                popt1, pcov = curve_fit(f, d, E, bounds=bounds, jac=self._model_jacobian, **kwargs)
            else: 
                popt1, pcov = curve_fit(f, d, E, bounds=bounds, **kwargs)
            E0, E1, logh, logC = popt1
            self.converged = True
        except RuntimeError:
            #print("\n\n*********\nFailed to fit single drug\n*********\n\n")
            E0 = np.max(E)
            E1 = np.min(E)
            logh = 0
            logC = np.log(np.median(d))
            self.converged = False

        
        
        self.E0 = E0
        self.Emax = E1
        self.h = np.exp(logh)
        self.C = np.exp(logC)

    # TODO: Implement Jacobian
    # TODO: Separate classes Hill2P() and Hill4P()
    def fit_2parameter(self, d, E, use_jacobian=True, **kwargs):
        if self.E0 is None: self.E0 = 1.
        if self.Emax is None: self.Emax = 0.

        f = lambda d, logh, logC: self._model(d, self.E0, self.Emax, np.exp(logh), np.exp(logC))

        bounds = tuple(zip(self.logh_bounds, self.logC_bounds))

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
            p0[0] = np.log(p0[0])
            p0[1] = np.log(p0[1])
            kwargs['p0'] = p0
            utils.sanitize_initial_guess(p0, bounds)
        else:
            p0 = [0, np.log(np.median(d))]
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0

        

        popt1, pcov = curve_fit(f, d, E, bounds=bounds, **kwargs)
        logh, logC = popt1
        self.h = np.exp(logh)
        self.C = np.exp(logC)


    def fit_CI(self, d, E):
        pass

    def E(self, d):
        if not self._is_parameterized():
            return 0
        return self._model(d, self.E0, self.Emax, self.h, self.C)

    def E_inv(self, E):
        if not self._is_parameterized():
            return 0
        return self._model_inv(E, self.E0, self.Emax, self.h, self.C)

    def get_parameters(self):
        return (self.E0, self.Emax, self.h, self.C)
        
    def _model(self, d, E0, Emax, h, C):
        dh = d**h
        return E0 + (Emax-E0)*dh/(C**h+dh)

    def _model_inv(self, E, E0, Emax, h, C):
        d = np.float_power((E0-Emax)/(E-Emax)-1.,1./h)*C
        return d

    def _model_jacobian(self, d, E0, Emax, logh, logC):
        dh = d**(np.exp(logh))
        Ch = (np.exp(logC))**(np.exp(logh))
        logd = np.log(d)

        jE0 = 1 - dh/(Ch+dh)
        jEmax = 1-jE0

        jC = (E0-Emax)*dh*np.exp(logh+logC)*(np.exp(logC))**(np.exp(logh)-1) / ((Ch+dh)*(Ch+dh))

        jh = (Emax-E0)*dh*np.exp(logh) * ((Ch+dh)*logd - (logC*Ch + logd*dh)) / ((Ch+dh)*(Ch+dh))
        
        return np.hstack((jE0.reshape(-1,1), jEmax.reshape(-1,1), jh.reshape(-1,1), jC.reshape(-1,1)))

    def _is_parameterized(self):
        return None not in (self.E0, self.Emax, self.h, self.C)

    def __repr__(self):
        return "Hill(%0.2f, %0.2f, %0.2f, %0.2e)"%(self.E0, self.Emax, self.h, self.C)
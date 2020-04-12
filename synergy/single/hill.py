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
            self.logh_bounds = (np.log10(h_bounds[0]), np.log10(h_bounds[1]))
            self.logC_bounds = (np.log10(C_bounds[0]), np.log10(C_bounds[1]))

        self.jacobian_queue = multiprocessing.Queue()
        self.converged = False
    
    def fit(self, d, E, use_jacobian=True, **kwargs):
        f = lambda d, E0, E1, logh, logC: self._model(d, E0, E1, 10.**logh, 10.**logC)

        bounds = tuple(zip(self.E0_bounds, self.Emax_bounds, self.logh_bounds, self.logC_bounds))

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
            p0[2] = np.log10(p0[2])
            p0[3] = np.log10(p0[3])
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0
        else:
            p0 = [max(E), min(E), 0, np.log10(np.median(d))]
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
            logC = np.log10(np.median(d))
            self.converged = False

        
        
        self.E0 = E0
        self.Emax = E1
        self.h = 10.**logh
        self.C = 10.**logC

    def fit_2parameter(self, d, E, use_jacobian=True, **kwargs):
        if self.E0 is None: self.E0 = 1.
        if self.Emax is None: self.Emax = 0.

        f = lambda d, logh, logC: self._model(d, self.E0, self.Emax, 10.**logh, 10.**logC)

        bounds = tuple(zip(self.logh_bounds, self.logC_bounds))

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
            p0[0] = np.log10(p0[0])
            p0[1] = np.log10(p0[1])
            kwargs['p0'] = p0
            utils.sanitize_initial_guess(p0, bounds)
        else:
            p0 = [0, np.log10(np.median(d))]
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0

        

        popt1, pcov = curve_fit(f, d, E, bounds=bounds, **kwargs)
        logh, logC = popt1
        self.h = 10.**logh
        self.C = 10.**logC


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

    def _model_jacobian_linear(self, d, E0, Emax, h, C):
        dh = d**h
        Ch = C**h
        logd = np.log(d)
        logC = np.log(C)

        jE0 = 1 - dh/(Ch+dh)
        jEmax = 1-jE0
        jC = (Emax-E0)*dh *(h*C**(h-1)) / (Ch+dh)**2
        jh = (Emax-E0) * ((Ch+dh)*logd*dh - dh*(logC*Ch + logd*dh)) / (Ch+dh)**2
        return np.asarray([jE0, jEmax, jh, jC]).transpose()

    def _model_jacobian(self, d, E0, Emax, logh, logC):
        dh = d**(10.**logh)
        Ch = (10.**logC)**(10.**logh)
        logd = np.log(d)
        ln10 = np.log(10)

        jE0 = 1 - dh/(Ch+dh)
        jEmax = 1-jE0

        jC = (E0-Emax)*dh*ln10*10.**(logh+logC)*(10.**logC)**(10.**logh-1) / ((Ch+dh)*(Ch+dh))

        jh = (Emax-E0)*dh*10.**logh*ln10 * ((Ch+dh)*logd - (logC*ln10*Ch + logd*dh)) / ((Ch+dh)*(Ch+dh))
        
        return np.hstack((jE0.reshape(-1,1), jEmax.reshape(-1,1), jh.reshape(-1,1), jC.reshape(-1,1)))

    @DeprecationWarning
    def _model_jacobian_parallel(self, d, E0, Emax, h, C):
        dh = d**(10.**h)
        Ch = (10.**C)**(10.**h)
        logd = np.log(d)
        ln10 = np.log(10)

        jobs = [multiprocessing.Process(target=func, args=(self.jacobian_queue, d, E0, Emax, h, C, dh, Ch, logd, ln10)) for func in (self._model_jacobian_E0, self._model_jacobian_Emax, self._model_jacobian_h, self._model_jacobian_C)]
        for job in jobs: job.start()
        for job in jobs: job.join()
        return np.hstack([self.jacobian_queue.get().reshape(-1,1) for job in jobs])

    @DeprecationWarning
    def _model_jacobian_E0(self, queue, d, E0, Emax, h, C, dh, Ch, logd, ln10):
        queue.put(1 - dh/(Ch+dh))

    @DeprecationWarning
    def _model_jacobian_Emax(self, queue, d, E0, Emax, h, C, dh, Ch, logd, ln10):
        queue.put(dh/(Ch+dh))

    @DeprecationWarning
    def _model_jacobian_h(self, queue, d, E0, Emax, h, C, dh, Ch, logd, ln10):
        queue.put((Emax-E0)*dh*10.**h*ln10 * ((Ch+dh)*logd - (C*ln10*Ch + logd*dh)) / ((Ch+dh)*(Ch+dh)))

    @DeprecationWarning
    def _model_jacobian_C(self, queue, d, E0, Emax, h, C, dh, Ch, logd, ln10):
        queue.put((E0-Emax)*dh*ln10*10.**(h+C)*(10.**C)**(10.**h-1) / ((Ch+dh)*(Ch+dh)))

    def _is_parameterized(self):
        return None not in (self.E0, self.Emax, self.h, self.C)

    def __repr__(self):
        return "Hill(%0.2f, %0.2f, %0.2f, %0.2e)"%(self.E0, self.Emax, self.h, self.C)
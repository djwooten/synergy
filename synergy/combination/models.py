import numpy as np
from scipy.optimize import curve_fit
import synergy.utils.utils as utils


class ModelNotParameterizedError(Exception):
    """
    The model must be parameterized prior to use. This can be done by calling
    fit(), or setParameters().
    """
    def __init__(self, msg='The model must be parameterized prior to use. This can be done by calling fit(), or setParameters().', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class Loewe:
    """
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf)):
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds

        with np.errstate(divide='ignore'):
            self.logh_bounds = (np.log10(h_bounds[0]), np.log10(h_bounds[1]))
            self.logC_bounds = (np.log10(C_bounds[0]), np.log10(C_bounds[1]))

        self._synergy = None
        self._drug1_model = None
        self._drug2_model = None
        
    
    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None):
        """
        """
        if drug1_model is None:
            mask = np.where(d2==min(d2))
            drug1_model = utils.fit_single(d1[mask], E[mask], self.E0_bounds, self.E1_bounds, self.h1_bounds, self.C1_bounds)
        if drug2_model is None:
            mask = np.where(d1==min(d1))
            drug2_model = utils.fit_single(d2[mask], E[mask], self.E0_bounds, self.E2_bounds, self.h2_bounds, self.C2_bounds)
        
        self._drug1_model = drug1_model
        self._drug2_model = drug2_model

        with np.errstate(divide='ignore', invalid='ignore'):
            d1_alone = drug1_model.E_inv(E)
            d2_alone = drug2_model.E_inv(E)

            self._synergy = d1/d1_alone + d2/d2_alone
        return self._synergy

    def null_E(d1, d2, drug1_model=None, drug2_model=None):
        if self._drug1_model is None or drug1_model is not None: self._drug1_model = drug1_model

        if self._drug2_model is None or drug2_model is not None: self._drug2_model = drug2_model

        if None in [self._drug1_model, self._drug2_model]:
            # Raise model not set error
            return 0

        with np.errstate(divide='ignore', invalid='ignore'):
            pass
        return 0

class Bliss:
    """
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf)):
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds
        self._synergy = None
        self._drug1_model = None
        self._drug2_model = None
        
    
    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None):
        """
        TODO: Add options for fitting ONLY using marginal points, not fits
        """
        if drug1_model is None:
            mask = np.where(d2==min(d2))
            drug1_model = utils.fit_single(d1[mask], E[mask], self.E0_bounds, self.E1_bounds, self.h1_bounds, self.C1_bounds)
        if drug2_model is None:
            mask = np.where(d1==min(d1))
            drug2_model = utils.fit_single(d2[mask], E[mask], self.E0_bounds, self.E2_bounds, self.h2_bounds, self.C2_bounds)
        
        self._drug1_model = drug1_model
        self._drug2_model = drug2_model

        E1_alone = drug1_model.E(d1)
        E2_alone = drug2_model.E(d2)
        self._synergy = E1_alone*E2_alone - E

        return self._synergy

    def null_E(self, d1, d2, drug1_model=None, drug2_model=None):
        if self._drug1_model is None or drug1_model is not None: self._drug1_model = drug1_model

        if self._drug2_model is None or drug2_model is not None: self._drug2_model = drug2_model

        if None in [self._drug1_model, self._drug2_model]:
            # Raise model not set error
            return 0

        D1, D2 = np.meshgrid(d1, d2)
        D1 = D1.flatten()
        D2 = D2.flatten()

        E1_alone = self._drug1_model.E(D1)
        E2_alone = self._drug2_model.E(D2)

        return D1, D2, E1_alone*E2_alone

class Zimmer:
    """
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf), a12_bounds=(-np.inf, np.inf), a21_bounds=(-np.inf, np.inf), h1=None, h2=None, C1=None, C2=None, a12=None, a21=None):
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.a12_bounds = a12_bounds
        self.a21_bounds = a21_bounds

        with np.errstate(divide='ignore'):
            self.logh1_bounds = (np.log10(h1_bounds[0]), np.log10(h1_bounds[1]))
            self.logC1_bounds = (np.log10(C1_bounds[0]), np.log10(C1_bounds[1]))
            self.logh2_bounds = (np.log10(h2_bounds[0]), np.log10(h2_bounds[1]))
            self.logC2_bounds = (np.log10(C2_bounds[0]), np.log10(C2_bounds[1]))

        self.h1 = h1
        self.h2 = h2
        self.C1 = C1
        self.C2 = C2
        self.a12 = a12
        self.a21 = a21

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):
        """
        
        """

        bounds = tuple(zip(self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.a12_bounds, self.a21_bounds))

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
            for i in range(4):
                p0[i] = np.log10(p0[i])
            utils.sanitize_initial_guess(p0, bounds)
            kwargs['p0'] = p0
        else:
            if drug1_model is None:
                mask = np.where(d2==min(d2))
                drug1_model = utils.fit_single_2parameter(d1[mask], E[mask], self.h1_bounds, self.C1_bounds, E0=1., Emax=0.)
            if drug2_model is None:
                mask = np.where(d1==min(d1))
                drug2_model = utils.fit_single_2parameter(d2[mask], E[mask], self.h2_bounds, self.C2_bounds, E0=1., Emax=0.)
            
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            p0 = [h1, h2, C1, C2, 0, 0]
            for i in range(4):
                p0[i] = np.log10(p0[i])
            bounds = tuple(zip(self.logh_bounds, self.logC_bounds))
            kwargs['p0']  = p0


        xdata = np.vstack((d1,d2))
        
        f = lambda d, logh1, logh2, logC1, logC2, a12, a21: self._model(d[0], d[1], 10.**logh1, 10.**logh2, 10.**logC1, 10.**logC2, a12, a21)
        
        popt1, pcov = curve_fit(f, xdata, E, bounds=bounds, **kwargs)

        logh1, logh2, logC1, logC2, a12, a21 = popt1
        self.h1 = 10.**logh1
        self.h2 = 10.**logh2
        self.C1 = 10.**logC1
        self.C2 = 10.**logC2
        self.a12 = a12
        self.a21 = a21
        return a12, a21

    def E(self, d1, d2):
        if None in [self.h1, self.h2, self.C1, self.C2, self.a12, self.a21]:
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

    def __repr__(self):
        return "Zimmer(h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, a12=%0.2f, a21=%0.2f)"%(self.h1, self.h2, self.C1, self.C2, self.a12, self.a21)
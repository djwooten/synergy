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

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit

from .. import utils
from ..utils import plots

class ParametricHigher(ABC):
    def __init__(self, parameters=None):
        self.bounds = None
        self.fit_function = None
        
        self.converged = False
        self.parameters = parameters

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
        self.bootstrap_parameters = None

    def _internal_fit(self, d, E, **kwargs):
        """Internal method to fit the model to data (d,E)
        """
        try:
            popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, **kwargs)
            return self._transform_params_from_fit(popt)
        except:
            return None

    def fit(self, d, E, bootstrap_iterations=0, **kwargs):

        #d = utils.remove_zeros_higher(d)
        d = np.asarray(d)
        E = np.asarray(E)

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
        else:
            p0 = None
        p0 = self._get_initial_guess(d, E, p0=p0)

        with np.errstate(divide='ignore', invalid='ignore'):
            popt = self._internal_fit(d, E, **kwargs)
        
        if popt is None:
            self.converged = False
            self.parametrers = _transform_params_from_fit(p0)

        else:
            self.converged = True
            self.parameters = popt
            self._score(d, E)
            kwargs['p0'] = self._transform_params_to_fit(popt)
            self._bootstrap_resample(d, E, bootstrap_iterations, **kwargs)

    def _bootstrap_resample(self, d, E, bootstrap_iterations, **kwargs):
        """Internal function to identify confidence intervals for parameters
        """

        if not self._is_parameterized(): return
        if not self.converged: return

        n_data_points = len(E)
        n_parameters = len(self.get_parameters())
        
        sigma_residuals = np.sqrt(self.sum_of_squares_residuals / (n_data_points - n_parameters))

        E_model = self.E(d)
        bootstrap_parameters = []

        for iteration in range(bootstrap_iterations):
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                popt1 = self._internal_fit(d, E_iteration, **kwargs)
            if popt1 is not None:
                bootstrap_parameters.append(popt1)

        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

    def _score(self, d, E):
        """Calculate goodness of fit and model quality scores, including sum-of-squares residuals, R^2, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

        If model is not yet paramterized, does nothing

        Called automatically during model.fit(d, E)

        Parameters
        ----------
        d : M x N
            Doses of N drugs sampled at M points
        
        E : array_like
            Dose-response at doses d
        """
        if (self._is_parameterized()):

            n_parameters = len(self.parameters)

            # TODO sum_of_squares_residuals for higher dimensional d
            self.sum_of_squares_residuals = utils.residual_ss(d, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    def _is_parameterized(self):
        """Returns False if any parameters are None or nan.

        Returns
        ----------
        is_parameterzed : bool
            True if all of the parameters are set. False if any are None or nan.
        """
        return not ((self.parameters is None) or (None in self.parameters) or (True in np.isnan(np.asarray(self.parameters))))

    @abstractmethod
    def _get_initial_guess(self, d, E, p0=None):
        """Internal method to format and/or guess p0
        """
        pass

    @abstractmethod
    def E(self, d):
        pass

    @staticmethod
    @abstractmethod
    def _transform_params_to_fit(params):
        pass

    @staticmethod
    @abstractmethod
    def _transform_params_from_fit(popt):
        pass

    @staticmethod
    @abstractmethod
    def _get_n_drugs_from_params(params):
        pass

    @abstractmethod
    def _get_initial_guess(self, d, E):
        pass
    
    @abstractmethod
    def _model(self, doses, *args):
        pass

    def plotly_isosurfaces(self, d, drug_axes=[0,1,2], other_drug_slices=None, **kwargs):
        if not self._is_parameterized():
            return None
        
        mask = d[:,0]>0
        n = d.shape[1]
        for i in range(n):
            if i in drug_axes:
                continue
            if other_drug_slices is None:
                dslice = np.min(d[:,i])
            else:
                dslice = other_drug_slices[i]
            mask = mask & (d[:,i]==dslice)

        E = self.E(d[mask,:])

        d1 = d[mask,drug_axes[0]]
        d2 = d[mask,drug_axes[1]]
        d3 = d[mask,drug_axes[2]]

        plots.plotly_isosurfaces(d1, d2, d3, E, **kwargs)

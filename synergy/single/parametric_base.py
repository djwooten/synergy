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

from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np
from .. import utils

class ParameterizedModel1D:
    def __init__(self):
        self.bounds = None
        self.fit_function = None
        self.jacobian_function = None
        
        self.converged = False
        self._fit = False

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
        self.bootstrap_parameters = None
    
    def _internal_fit(self, d, E, use_jacobian, verbose=True, **kwargs):
        """Internal method to fit the model to data (d,E)
        """
        try:
        #if True:
            if use_jacobian:
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, jac=self.jacobian_function, **kwargs)
            else: 
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, **kwargs)
            if True in np.isnan(popt):
                return None
            return self._transform_params_from_fit(popt)
        except Exception as err:
            if verbose:
                print("Exception during single drug response fit: %s"%err)
            return None

    def _get_initial_guess(self, d, E, p0=None):
        """Internal method to format and/or guess p0
        """
        return p0

    def _set_parameters(self, popt):
        """Internal method to set model parameters
        """
        pass

    def fit(self, d, E, use_jacobian=True, bootstrap_iterations=0, bootstrap_confidence_interval=95, **kwargs):
        """Fit the Hill equation to data. Fitting algorithm searches for h and C in a log-scale, but all bounds and guesses should be provided in a linear scale.

        Parameters
        ----------
        d : array_like
            Array of doses measured
        
        E : array_like
            Array of effects measured at doses d
        
        use_jacobian : bool, default=True
            If True, will use the Jacobian to help guide fit. When the number
            of data points is less than a few hundred, this makes the fitting
            slower. However, it also improves the reliability with which a fit
            can be found.
        
        kwargs
            kwargs to pass to scipy.optimize.curve_fit()
        """

        self._fit = True
        d = np.asarray(d)
        E = np.asarray(E)

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
        else:
            p0 = None
        p0 = self._get_initial_guess(d, E, p0=p0)
        kwargs['p0']=p0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            popt = self._internal_fit(d, E, use_jacobian, **kwargs)

        if popt is None:
            self.converged = False
            self._set_parameters(self._transform_params_from_fit(p0))
        else:
            self.converged = True
            self._set_parameters(popt)

            n_parameters = len(popt)
            n_samples = len(d)
            if (n_samples - n_parameters - 1 > 0):
                self._score(d, E)
                kwargs['p0'] = self._transform_params_to_fit(popt)
                self._bootstrap_resample(d, E, use_jacobian, bootstrap_iterations, bootstrap_confidence_interval, **kwargs)

    def E(self, d):
        """Evaluate this model at dose d.

        Parameters
        ----------
        d : array_like
            Doses to calculate effect at
        
        Returns
        ----------
        effect : array_like
            Evaluate's the model at dose in d
        """
        ret = 0*d
        ret[:] = np.nan
        return ret
        
    def E_inv(self, E):
        """Evaluate the inverse of this model.

        Parameters
        ----------
        E : array_like
            Effects to get the doses for
        
        Returns
        ----------
        doses : array_like
            Doses which achieve effects E using this model. Will return np.nan for effects outside of the model's effect range, or for non-invertable models
        """
        ret = 0*d
        ret[:] = np.nan
        return ret

    def _transform_params_from_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform those to h and C
        """
        return params

    def _transform_params_to_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform from h and C
        """
        return params

    def get_parameters(self):
        """Returns model parameters
        """
        return []

    def get_parameter_range(self, confidence_interval=95):
        """Returns the lower bound and upper bound estimate for each parameter.

        Parameters:
        -----------
        confidence_interval : int, float, default=95
            % confidence interval to return. Must be between 0 and 100.
        """
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
        return np.percentile(self.bootstrap_parameters, [lb, ub], axis=0)

    def _is_parameterized(self):
        """Internalized method to check all model parameters are set
        """
        return not (None in self.get_parameters() or True in np.isnan(np.asarray(self.get_parameters())))

    def _score(self, d, E):
        """Calculate goodness of fit and model quality scores, including sum-of-squares residuals, R^2, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

        If model is not yet paramterized, does nothing

        Called automatically during model.fit(d1, d2, E)

        Parameters
        ----------
        d : array_like
            Doses
        
        E : array_like
            Measured dose-response at doses d
        """
        if (self._is_parameterized()):

            n_parameters = len(self.get_parameters())

            self.sum_of_squares_residuals = utils.residual_ss_1d(d, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, confidence_interval, **kwargs):
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
            with np.errstate(divide='ignore', invalid='ignore'):
                popt1 = self._internal_fit(d, E_iteration, verbose=False, use_jacobian=use_jacobian, **kwargs)
            
            if popt1 is not None:
                bootstrap_parameters.append(popt1)
        
        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

    def is_fit(self):
        return self._is_parameterized()
        #return self._fit
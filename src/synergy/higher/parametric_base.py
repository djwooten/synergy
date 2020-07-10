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
from scipy.stats import norm

from .. import utils
from ..utils import plots

class ParametricHigher(ABC):
    """The abstract base class for higher dimensional (3+ drug) parametric synergy models.
    """
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
        except Exception as e:
            print("Exception:", e)
            return None

    def fit(self, d, E, bootstrap_iterations=0, **kwargs):
        """Fits the model to data

        Parameters
        ----------
        d : numpy.ndarray (M x N)
            Doses of N drugs sampled at M points
        
        E : array_like
            Dose-response at doses d

        bootstrap_iterations : int , default=0
            The number of boostrap resample iterations performed to estimate parameter confidence intervals.

        kwargs
            kwargs to pass to scipy.optimize.curve_fit()
        """
        d = np.asarray(d)
        E = np.asarray(E)

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
        else:
            p0 = None
        p0 = self._get_initial_guess(d, E, p0=p0)
        kwargs['p0'] = p0

        with np.errstate(divide='ignore', invalid='ignore'):
            popt = self._internal_fit(d, E, **kwargs)
        
        if popt is None:
            self.converged = False
            self.parameters = self._transform_params_from_fit(p0)

        else:
            self.converged = True
            self.parameters = popt
            n_parameters = len(popt)
            n_samples = d.shape[0]
            if (n_samples - n_parameters - 1 > 0):
                self._score(d, E)
                kwargs['p0'] = self._transform_params_to_fit(popt)
                self._bootstrap_resample(d, E, bootstrap_iterations, **kwargs)
            

    def _bootstrap_resample(self, d, E, bootstrap_iterations, **kwargs):
        """Internal function to identify confidence intervals for parameters
        """

        if not self._is_parameterized(): return
        if not self.converged: return

        n_data_points = len(E)
        n_parameters = len(self.parameters)
        
        sigma_residuals = np.sqrt(self.sum_of_squares_residuals / (n_data_points - n_parameters))

        E_model = self.E(d)
        bootstrap_parameters = []

        for iteration in range(bootstrap_iterations):
            print("Bootstrap iteration = %d"%iteration)
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
        d : numpy.ndarray (M x N)
            Doses of N drugs sampled at M points
        
        E : array_like
            Dose-response at doses d
        """
        if (self._is_parameterized()):

            n_parameters = len(self.parameters)

            self.sum_of_squares_residuals = utils.residual_ss_1d(d, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

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
        """Returns False if any parameters are None or nan.

        Returns
        ----------
        is_parameterzed : bool
            True if all of the parameters are set. False if any are None or nan.
        """
        return not ((self.parameters is None) or (None in self.parameters) or (True in np.isnan(np.asarray(self.parameters))))

    @abstractmethod
    def _get_initial_guess(self, d, E, p0=None):
        """Internal method to format and/or come up with initial guess for parameters
        """
        pass

    @abstractmethod
    def _get_single_drug_classes(self):
        """
        Returns
        -------
        default_single_class : class
            The default class type to use for single-drug models

        expected_single_superclass : class
            The required type for single-drug models. If a single-drug model is passed that is not an instance of this superclass, it will be re-instantiated using default_model
        """
        pass

    @abstractmethod
    def E(self, d):
        """Evaluates the model at dose d

        Parameters
        ----------
        d : numpy.ndarray
            Doses, in an M x N ndarray, where M is the number of samples, and N is the number of drugs.

        Returns
        ----------
        E : numpy.array
            Model effect evaluated at doses d.
            
        """
        pass

    @abstractmethod
    def get_parameters(self, confidence_interval=95):
        """Returns a dict of the model's parameters.

        When relevant, it will also return meaningful derived parameters. For instance, MuSyC has several parameters for E, but defines a synergy parameter beta as a function of E parameters. Thus, beta will also be included.
        
        If the model was fit to data with bootstrap_iterations > 0, this will also return the specified confidence interval.
        """
        pass

    @abstractmethod
    def summary(self, confidence_interval=95, tol=0.01):
        """Summarizes the model's synergy conclusions.

        For each synergy parameters, determines whether it indicates synergy or antagonism. When the model has been fit with bootstrap_parameters>0, the best fit, lower bound, and upper bound must all agree on synergy or antagonism.
    
        Parameters
        ----------
        confidence_interval : float, optional (default=95)
            If the model was fit() with bootstrap_parameters>0, confidence_interval will be used to get the upper and lower bounds. 

        tol : float, optional (default=0.01)
            Tolerance to determine synergy or antagonism. The parameter must exceed the threshold by at least tol (some parameters, like MuSyC's alpha which is antagonistic from 0 to 1, and synergistic from 1 to inf, will be log-scaled prior to comparison with tol)

        Returns
        ----------
        summary : str
            Tab-separated string. If the model has been bootstrapped, columns are [parameter, value, (lower,upper), synergism/antagonism]. If the model has not been bootstrapped, columns are [parameter, value, synergism/antagonism].
        """
        pass

    @abstractmethod
    def _transform_params_to_fit(self, params):
        """Some parameters may be fit on nonlinear (e.g., log) scales. This method transforms linear parameters into this scale.
        """
        pass

    @abstractmethod
    def _transform_params_from_fit(self, popt):
        """Some parameters may be fit on nonlinear (e.g., log) scales. These transformed-parameters must be transformed back to a linear scale.
        """
        pass

    @abstractmethod
    def _get_n_drugs_from_params(self, params):
        """Determine the number of drugs that are being modeled, given the size of the params array.
        """
        pass

    @abstractmethod
    def _get_initial_guess(self, d, E, p0=None):
        """Gets the initial guess used for curve_fitting
        """
        pass
    
    @abstractmethod
    def _model(self, doses, *args):
        """Model for higher dimensional parametric synergy models.

        Parameters
        ----------
        doses : numpy.ndarray
            M x N ndarray, where M is the number of samples, and N is the number of drugs.

        args
            Parameters for the model.
        """
        pass

    def plotly_isosurfaces(self, d, drug_axes=[0,1,2], other_drug_slices=None, cmap="YlGnBu", **kwargs):
        if not self._is_parameterized():
            return None
        
        d = np.asarray(d)
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

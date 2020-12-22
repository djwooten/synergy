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
from ..utils import plots, dose_tools



class ParametricModel(ABC):
    """Base class for paramterized synergy models, including MuSyC, Zimmer, GPDI, and BRAID.
    """
    def __init__(self):
        """Bounds for drug response parameters (for instance, given percent viability data, one might expect E to be bounded within (0,1)) can be set, or parameters can be explicitly set.
        """

        self.bounds = None
        self.fit_function = None
        self.jacobian_function = None
        
        self.converged = False

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
        self.bootstrap_parameters = None

    def _score(self, d1, d2, E):
        """Calculate goodness of fit and model quality scores, including sum-of-squares residuals, R^2, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

        If model is not yet paramterized, does nothing

        Called automatically during model.fit(d1, d2, E)

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        E : array_like
            Dose-response at doses d1 and d2
        """
        if (self._is_parameterized()):

            n_parameters = len(self._get_parameters())

            self.sum_of_squares_residuals = utils.residual_ss(d1, d2, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    @abstractmethod
    def _get_parameters(self):
        """Returns all of the model's fit parameters

        Returns
        ----------
        parameters : list or tuple
            Model's parameters
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

    def _internal_fit(self, d, E, use_jacobian, verbose=True, **kwargs):
        """Internal method to fit the model to data (d,E)
        """
        try:
            if use_jacobian and self.jacobian_function is not None:
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, jac=self.jacobian_function, **kwargs)
            else: 
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, **kwargs)
            if True in np.isnan(popt):
                return None
            return self._transform_params_from_fit(popt)
        except Exception as err:
            if verbose:
                print("Exception during combination drug response fit: %s"%err)
            return None

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian = True, p0=None, bootstrap_iterations=0, seed=None, **kwargs):
        """Fit the model to data.

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2

        E : array_like
            Dose-response at doses d1 and d2

        drug1_model : single-drug-model, default=None
            Only used when p0 is None. Pre-defined, or fit, model (e.g., Hill()) of drug 1 alone. Parameters from this model are used to provide an initial guess of E0, E1, h1, and C1 for the 2D-model fit. If None (and p0 is None), then d1 and E will be masked where d2==min(d2), and used to fit a model for drug 1.

        drug2_model : single-drug-model, default=None
            Same as drug1_model, for drug 2.
        
        use_jacobian : bool, default=True
            If True, will use the Jacobian to help guide fit (ONLY MuSyC, Hill, and Hill_2P have Jacobian implemented yet). When the number
            of data points is less than a few hundred, this makes the fitting
            slower. However, it also improves the reliability with which a fit
            can be found. If drug1_model or drug2_model are None, use_jacobian will also be applied for their fits.

        p0 : tuple, default=None
            Initial guess for the parameters. If p0 is None (default), drug1_model and drug2_model will be used to obtain an initial guess. If they are also None, they will be fit to the data. If they fail to fit, the initial guess will be E0=max(E), Emax=min(E), h=1, C=median(d), and all synergy parameters are additive (i.e., at the boundary between antagonistic and synergistic)

        seed : int, default=None
            If not None, used as numpy.random.seed(start_seed) at the beginning of bootstrap resampling
        
        kwargs
            kwargs to pass to scipy.optimize.curve_fit()
        """

        if seed is not None: np.random.seed(seed)
        d1 = np.asarray(d1, dtype=np.float64)
        d2 = np.asarray(d2, dtype=np.float64)

        E = np.asarray(E)

        xdata = np.vstack((d1,d2))
        
        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
        else:
            p0 = None
        
        p0 = self._get_initial_guess(d1, d2, E, drug1_model, drug2_model, p0=p0)

        kwargs['p0']=p0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            popt = self._internal_fit(xdata, E, use_jacobian, **kwargs)

        if popt is None:
            self._set_parameters(self._transform_params_from_fit(p0))
            self.converged = False
        else:
            self.converged = True
            self._set_parameters(popt)
            n_parameters = len(popt)
            n_samples = len(d1)
            if (n_samples - n_parameters - 1 > 0):
                self._score(d1, d2, E)
                kwargs['p0'] = self._transform_params_to_fit(popt)
                self._bootstrap_resample(d1, d2, E, use_jacobian, bootstrap_iterations, **kwargs)
    
    @abstractmethod
    def E(self, d1, d2):
        """Returns drug effect E at dose d1,d2 for a pre-defined or fitted model.

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        Returns
        ----------
        effect : array_like
            Evaluate's the model at doses d1 and d2
        """
        pass

    def _is_parameterized(self):
        """Returns False if any parameters are None or nan.

        Returns
        ----------
        is_parameterzed : bool
            True if all of the parameters are set. False if any are None or nan.
        """
        return not (None in self._get_parameters() or True in np.isnan(np.asarray(self._get_parameters())))

    @abstractmethod
    def _set_parameters(self, popt):
        """Internal method to set model parameters
        """
        pass

    @abstractmethod
    def _transform_params_from_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform those to h and C
        """
        pass

    @abstractmethod
    def _transform_params_to_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform from h and C
        """
        pass

    @abstractmethod
    def _get_initial_guess(self, d1, d2, E, drug1_model, drug2_model, p0=None):
        """Internal method to format and/or guess p0
        """
        pass

    
    def _bootstrap_resample(self, d1, d2, E, use_jacobian, bootstrap_iterations, seed=None, **kwargs):
        """Internal function to identify confidence intervals for parameters
        """

        if not self._is_parameterized(): return
        if not self.converged: return

        n_data_points = len(E)
        n_parameters = len(self._get_parameters())
        
        sigma_residuals = np.sqrt(self.sum_of_squares_residuals / (n_data_points - n_parameters))

        E_model = self.E(d1, d2)
        bootstrap_parameters = []

        xdata = np.vstack((d1,d2))

        #if start_seed is not None: np.random.seed(start_seed)
        for iteration in range(bootstrap_iterations):
            #if start_seed is not None: np.random.seed(start_seed + iteration)
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                popt1 = self._internal_fit(xdata, E_iteration, verbose=False, use_jacobian=use_jacobian, **kwargs)
            
            if popt1 is not None:
                bootstrap_parameters.append(popt1)
        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

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

    def plot_heatmap(self, d1, d2, cmap="YlGnBu", **kwargs):
        """Plots the model's effect, E(d1, d2) as a heatmap

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        kwargs
            kwargs passed to synergy.utils.plots.plot_heatmap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        E = self.E(d1, d2)
        plots.plot_heatmap(d1, d2, E, cmap=cmap, **kwargs)

    def plot_residual_heatmap(self, d1, d2, E, cmap="RdBu", center_on_zero=True, **kwargs):
        """Plots the residuals of the fit model as a heatmap

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2

        E : array_like
            Observed drug effects

        cmap : string, default="RdBu"
            Colormap for the plot
        
        kwargs
            kwargs passed to synergy.utils.plots.plot_heatmap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        Emodel = self.E(d1, d2)
        plots.plot_heatmap(d1, d2, E-Emodel, cmap=cmap, center_on_zero=center_on_zero, **kwargs)

    @abstractmethod
    def _reference_E(self, d1, d2):
        pass

    def plot_reference_heatmap(self, d1, d2, cmap="YlGnBu", **kwargs):
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return

        Ereference = self._reference_E(d1, d2)
        plots.plot_heatmap(d1, d2, Ereference, cmap=cmap, **kwargs)

    def plot_reference_surface(self, d1, d2, cmap="YlGnBu", **kwargs):
        if not self._is_parameterized():
            return
        Ereference = self._reference_E(d1, d2)
        plots.plot_surface_plotly(d1, d2, Ereference, cmap=cmap, **kwargs)

    def plot_delta_heatmap(self, d1, d2, cmap="PRGn", center_on_zero=True, **kwargs):
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        Ereference = self._reference_E(d1, d2)
        Emodel = self.E(d1, d2)
        plots.plot_heatmap(d1, d2, Ereference-Emodel, cmap=cmap, center_on_zero=center_on_zero, **kwargs)

    def plot_delta_surface(self, d1, d2, cmap="PRGn", center_on_zero=True, **kwargs):
        if not self._is_parameterized():
            return
        Ereference = self._reference_E(d1, d2)
        Emodel = self.E(d1, d2)
        plots.plot_surface_plotly(d1, d2, Ereference-Emodel, cmap=cmap, center_on_zero=center_on_zero, **kwargs)
    

    def plot_surface_plotly(self, d1, d2, cmap="YlGnBu", **kwargs):
        """Plots the model's effect, E(d1, d2) as a surface using synergy.utils.plots.plot_surface_plotly()

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        cmap : string, default="viridis"
            Colorscale for the plot

        kwargs
            kwargs passed to synergy.utils.plots.plot_heatmap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        # d1 and d2 may come from data, and have replicates. This would cause problems with surface plots (replicates in scatter_points are fine, but replicates in the surface itself are not)
        d1, d2 = dose_tools.remove_replicates(d1, d2)
        E = self.E(d1, d2)
        plots.plot_surface_plotly(d1, d2, E, cmap=cmap, **kwargs)
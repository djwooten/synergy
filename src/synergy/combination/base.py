"""
    Copyright (C) 2020 David J. Wooten

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from .. import utils
from ..utils import plots


class ParameterizedModel:
    """
    This is the base class for paramterized synergy models, including MuSyC, Zimmer, GPDI, and BRAID.
    """
    def __init__(self):
        """Bounds for drug response parameters (for instance, given percent viability data, one might expect E to be bounded within (0,1)) can be set, or parameters can be explicitly set.
        """

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
#        self.converged = False

    def _score(self, d1, d2, E):
        """Calculate goodness of fit and model quality scores, including sum-of-squares residuals, R^2, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

        If model is not yet paramterized, does nothing

        Called automatically during model.fit(d1, d2, E)

        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2
        
        E : array-like
            Dose-response at doses d1 and d2
        """
        if (self._is_parameterized()):

            n_parameters = len(self.get_parameters())

            self.sum_of_squares_residuals = utils.residual_ss(d1, d2, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    def get_parameters():
        """
        Returns
        ----------
        parameters : list or tuple
            Model's parameters
        """
        return []

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian = True, p0=None, **kwargs):
        """Fit the synergy model to data using scipy.optimize.curve_fit().

        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2

        E : array-like
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
        
        kwargs
            kwargs to pass to scipy.optimize.curve_fit()

        Returns
        ----------
        synergy_parameters : array-like
            The fit parameters describing the synergy in the data
        """

    def E(self, d1, d2):
        """
        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2
        
        Returns
        ----------
        effect : array-like
            Evaluate's the model at doses d1 and d2
        """
        return 0

    def _is_parameterized(self):
        """
        Returns
        ----------
        is_parameterzed : bool
            True if all of the parameters are set. False if any are None or nan.
        """
        return not (None in self.get_parameters() or True in np.isnan(np.asarray(self.get_parameters())))

    def plot_colormap(self, d1, d2, cmap="viridis", **kwargs):
        """Plots the model's effect, E(d1, d2) as a heatmap

        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2
        
        kwargs
            kwargs passed to synergy.utils.plots.plot_colormap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        E = self.E(d1, d2)
        plots.plot_colormap(d1, d2, E, cmap=cmap, **kwargs)

    def plot_surface_plotly(self, d1, d2, cmap="RdBu", **kwargs):
        """Plots the model's effect, E(d1, d2) as a surface using synergy.utils.plots.plot_surface_plotly()

        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2
        
        kwargs
            kwargs passed to synergy.utils.plots.plot_colormap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        E = self.E(d1, d2)
        plots.plot_surface_plotly(d1, d2, E, cmap=cmap, **kwargs)

    def create_fit(d1, d2, E, **kwargs):
        """Courtesy one-liner to create a model and fit it to data. Appropriate (see model __init__() for details) bounds may be set for curve_fit.

        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2
        
        E : array-like
            Dose-response at doses d1 and d2

        X_bounds : tuple, 
            Bounds for each parameter of the model to be fit. See model.__init__() for specific details.
        
        kwargs
            kwargs to pass sto model.fit()

        Returns
        ----------
        model : ParametricModel
            Synergy model fit to the given data
        """
        return


class DoseDependentModel:
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf)):
        """
        Parameters
        ----------
        E0_bounds: tuple, default=(-np.inf, np.inf)
            Bounds to use for E0 to fit drug1_model and drug2_model if they are not supplied directly in .fit()
        
        E{X}_bounds: tuple, default=(-np.inf, np.inf)
            Bounds to use for Emax to fit drug{X}_model if it is not supplied directly in .fit() (e.g., E1_bounds will constrain drug 1's Emax)
        
        h{X}_bounds: tuple, default=(0, np.inf)
            Bounds to use for hill slope to fit drug{X}_model if it is not supplied directly in .fit()
        
        C{X}_bounds: tuple, default=(0, np.inf)
            Bounds to use for EC50 to fit drug{X}_model if it is not supplied directly in .fit()
        """
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds

        self.synergy = None
        self.d1 = None
        self.d2 = None

        self.drug1_model = None
        self.drug2_model = None

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):
        """Calculates dose-dependent synergy at doses d1, d2.

        Parameters
        ----------
        d1 : array-like
            Doses of drug 1
        
        d2 : array-like
            Doses of drug 2

        E : array-like
            Dose-response at doses d1 and d2

        drug1_model : single-drug-model, default=None
            Pre-defined, or fit, model (e.g., Hill()) of drug 1 alone. If None (default), then d1 and E will be masked where d2==min(d2), and used to fit a model (Hill, Hill_2P, or Hill_CI, depending on the synergy model) for drug 1.

        drug2_model : single-drug-model, default=None
            Same as drug1_model, for drug 2.
        
        kwargs
            kwargs to pass to Hill.fit() (or whichever single-drug model is used)

        Returns
        ----------
        synergy : array-like
            The synergy calculated at all doses d1, d2
        """
        self.d1 = d1
        self.d2 = d2
        self.synergy = 0*d1
        self.synergy[:] = np.nan
        return self.synergy

    def plot_colormap(self, cmap="PRGn", neglog=False, **kwargs):
        """Plots the synergy as a heatmap

        Parameters
        ----------
        cmap : string, default="PRGn"
            Colorscale for the plot

        neglog : bool, default=False
            If True, will transform the synergy values by -log(synergy). Loewe and CI are synergistic between [0,1) and antagonistic between (1,inf). Thus, -log(synergy) becomes synergistic for positive values, and antagonistic for negative values. This behavior matches other synergy frameworks, and helps better visualize results. But it is never set by default.

        kwargs
            kwargs passed to synergy.utils.plots.plot_colormap()
        """
        if neglog:
            with np.errstate(invalid="ignore"):
                plots.plot_colormap(self.d1, self.d2, -np.log(self.synergy), cmap=cmap, **kwargs)
        else:
            plots.plot_colormap(self.d1, self.d2, self.synergy, cmap=cmap, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", **kwargs):
        """Plots the synergy as a 3D surface using synergy.utils.plots.plot_surface_plotly()

        Parameters
        ----------
        kwargs
            kwargs passed to synergy.utils.plots.plot_colormap()
        """
        plots.plot_surface_plotly(self.d1, self.d2, self.synergy, cmap=cmap, **kwargs)

class ModelNotParameterizedError(Exception):
    """
    The model must be parameterized prior to use. This can be done by calling
    fit(), or setParameters().
    """
    def __init__(self, msg='The model must be parameterized prior to use. This can be done by calling fit(), or setParameters().', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class FeatureNotImplemented(Warning):
    """
    """
    def __init__(self, msg="This feature is not yet implemented", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

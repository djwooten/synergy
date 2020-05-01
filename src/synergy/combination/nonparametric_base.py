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
from ..utils import plots


class DoseDependentModel:
    """These are models for which synergy is defined independently at each individual dose.
    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf)):
        """Creates a DoseDependentModel

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
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2

        E : array_like
            Dose-response at doses d1 and d2

        drug1_model : single-drug-model, default=None
            Pre-defined, or fit, model (e.g., Hill()) of drug 1 alone. If None (default), then d1 and E will be masked where d2==min(d2), and used to fit a model (Hill, Hill_2P, or Hill_CI, depending on the synergy model) for drug 1.

        drug2_model : single-drug-model, default=None
            Same as drug1_model, for drug 2.
        
        kwargs
            kwargs to pass to Hill.fit() (or whichever single-drug model is used)

        Returns
        ----------
        synergy : array_like
            The synergy calculated at all doses d1, d2
        """
        self.d1 = d1
        self.d2 = d2
        self.synergy = 0*d1
        self.synergy[:] = np.nan
        return self.synergy

    def plot_colormap(self, cmap="PRGn", neglog=False, center_on_zero=True, **kwargs):
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
                plots.plot_colormap(self.d1, self.d2, -np.log(self.synergy), cmap=cmap, center_on_zero=True, **kwargs)
        else:
            plots.plot_colormap(self.d1, self.d2, self.synergy, cmap=cmap, center_on_zero=True, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", **kwargs):
        """Plots the synergy as a 3D surface using synergy.utils.plots.plot_surface_plotly()

        Parameters
        ----------
        kwargs
            kwargs passed to synergy.utils.plots.plot_colormap()
        """
        plots.plot_surface_plotly(self.d1, self.d2, self.synergy, cmap=cmap, **kwargs)
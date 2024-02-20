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

from synergy.utils import base as utils
from synergy.exceptions import ModelNotParameterizedError
from synergy.single import LogLinear
from synergy.utils import plots


class DoseDependentModel(ABC):
    """These are models for which synergy is defined independently at each individual dose."""

    def __init__(self, drug1_model=None, drug2_model=None, **kwargs):
        """Ctor."""
        self.synergy = None
        self.d1 = None
        self.d2 = None
        self.reference = None

        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class

        self.drug1_model = utils.sanitize_single_drug_model(drug1_model, default_type, required_type, **kwargs)
        self.drug2_model = utils.sanitize_single_drug_model(drug2_model, default_type, required_type, **kwargs)

    def fit(self, d1, d2, E, **kwargs):
        """Calculates dose-dependent synergy at doses d1, d2.

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1

        d2 : array_like
            Doses of drug 2

        E : array_like
            Dose-response at doses d1 and d2

        kwargs
            kwargs to pass to model.fit() for the single-drug models

        Returns
        ----------
        synergy : array_like
            The synergy calculated at all doses d1, d2
        """
        self.d1 = d1
        self.d2 = d2
        self.synergy = d1 * np.nan

        # Fit the single drug models if they were not pre-fit by the user
        if not self.drug1_model.is_specified:
            mask = np.where(d2 == min(d2))
            self.drug1_model.fit(d1[mask], E[mask], **kwargs)

        if not self.drug2_model.is_specified:
            mask = np.where(d1 == min(d1))
            self.drug2_model.fit(d2[mask], E[mask], **kwargs)

        if not self.is_specified:
            raise ModelNotParameterizedError("Cannot calculate synergy because the model is not specified")

        self.reference = self._E_reference(d1, d2)
        self.synergy = self._get_synergy(d1, d2, E)

        return self.synergy

    @abstractmethod
    def _E_reference(self, d1, d2):
        """Calculate the reference surface"""

    @abstractmethod
    def _get_synergy(self, d1, d2, E):
        """Calculate synergy"""

    def _sanitize_synergy(self, d1, d2, synergy, default_val: float):
        if hasattr(synergy, "__iter__"):
            synergy[(d1 == 0) | (d2 == 0)] = default_val
        elif d1 == 0 or d2 == 0:
            synergy = default_val
        return synergy

    @property
    def is_specified(self):
        return self.drug1_model.is_specified and self.drug2_model.is_specified

    @property
    def _default_single_drug_class(self) -> type:
        """The default drug model to use"""
        return LogLinear

    @property
    def _required_single_drug_class(self) -> type:
        """The required superclass of the models for the individual drugs, or None if any model is acceptable"""
        return None

    def plot_heatmap(self, cmap="PRGn", neglog=False, center_on_zero=True, **kwargs):
        """Plots the synergy as a heatmap

        Parameters
        ----------
        cmap : string, default="PRGn"
            Colorscale for the plot

        neglog : bool, default=False
            If True, will transform the synergy values by -log(synergy). Loewe and CI are synergistic between [0,1) and antagonistic between (1,inf). Thus, -log(synergy) becomes synergistic for positive values, and antagonistic for negative values. This behavior matches other synergy frameworks, and helps better visualize results. But it is never set by default.

        center_on_zero : bool, default=True
            If True, will set vmin and vmax to be centered around 0.

        kwargs
            kwargs passed to synergy.utils.plots.plot_heatmap()
        """
        if neglog:
            with np.errstate(invalid="ignore"):
                plots.plot_heatmap(
                    self.d1,
                    self.d2,
                    -np.log(self.synergy),
                    cmap=cmap,
                    center_on_zero=center_on_zero,
                    **kwargs,
                )
        else:
            plots.plot_heatmap(self.d1, self.d2, self.synergy, cmap=cmap, center_on_zero=center_on_zero, **kwargs)

    def plot_reference_heatmap(self, cmap="YlGnBu", **kwargs):
        if self.reference is not None:
            plots.plot_heatmap(self.d1, self.d2, self.reference, cmap=cmap, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", neglog=False, **kwargs):
        """Plots the synergy as a 3D surface using synergy.utils.plots.plot_surface_plotly()

        Parameters
        ----------
        cmap : string, default="PRGn"
            Colorscale for the plot

        neglog : bool, default=False
            If True, will transform the synergy values by -log(synergy). Loewe and CI are synergistic between [0,1) and antagonistic between (1,inf). Thus, -log(synergy) becomes synergistic for positive values, and antagonistic for negative values. This behavior matches other synergy frameworks, and helps better visualize results. But it is never set by default.

        kwargs
            kwargs passed to synergy.utils.plots.plot_heatmap()
        """
        if neglog:
            with np.errstate(invalid="ignore"):
                plots.plot_surface_plotly(self.d1, self.d2, -np.log(self.synergy), cmap=cmap, **kwargs)
        else:
            plots.plot_surface_plotly(self.d1, self.d2, self.synergy, cmap=cmap, **kwargs)

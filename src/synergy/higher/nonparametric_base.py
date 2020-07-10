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
from abc import ABC, abstractmethod
import inspect


from ..utils import plots
from .. import utils


class DoseDependentHigher(ABC):
    """These are models for which synergy is defined independently at each individual dose.
    """
    def __init__(self, E_bounds=(-np.inf,np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf)):
        """Creates a DoseDependentModel

        """
        self.synergy = None
        self.d = None
        self.single_models = None

        self.E_bounds = E_bounds
        self.h_bounds = h_bounds
        self.C_bounds = C_bounds

    def fit(self, d, E, single_models=None, **kwargs):
        """Calculates dose-dependent synergy at doses d1, d2.

        Parameters
        ----------
        d : numpy.ndarray (M x N)
            Doses of N drugs sampled at M points
        
        E : array_like with length equal to M
            Dose-response at doses d

        single_models : class, array_like with length equal to N
            Model(s) to use for single drugs. If a class is given, that class is used to instantiate models for each drug. If an array is given, each element of that array must be a model that will be used for the corresponding drug.
        
        kwargs
            kwargs to pass to Hill.fit() (or whichever single-drug model is used)

        Returns
        ----------
        synergy : array_like
            The synergy calculated at all doses d
        """
        self.d = d
        self.synergy = 0*E
        self.synergy[:] = np.nan

        N = d.shape[1]

        # Initialize single drug models
        
        default_class, expected_superclass = self._get_single_drug_classes()
        
        # If an array is given, sanitize each element of the array
        if hasattr(single_models, "__iter__"):
            self.single_models = [utils.sanitize_single_drug_model(_model, default_class, expected_superclass=expected_superclass, E0_bounds=self.E_bounds, Emax_bounds=self.E_bounds, h_bounds=self.h_bounds, C_bounds=self.C_bounds) for _model in single_models]

        # Otherwise, run sanitize N times, generating N models to use
        else:
            self.single_models = []
            for _i in range(N):
                self.single_models.append(utils.sanitize_single_drug_model(single_models, default_class, expected_superclass=expected_superclass, E0_bounds=self.E_bounds, Emax_bounds=self.E_bounds, h_bounds=self.h_bounds, C_bounds=self.C_bounds))
             
        # Fit all single models if they were not given pre-fit
        for i in range(N):
            single = self.single_models[i]
            if not single.is_fit():
                # Mask where all other drugs are minimum (ideally 0)
                mask = d[:,i]>=0 # This should always be true
                for j in range(N):
                    if i==j: continue
                    mask = mask & (d[:,j]==np.min(d[:,j]))
                mask = np.where(mask)
                
                single.fit(d[mask,i].flatten(), E[mask], **kwargs)

        return self.synergy

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

    def plotly_isosurfaces(self, drug_axes=[0,1,2], other_drug_slices=None, cmap="PRGn", neglog=False, **kwargs):
        mask = self.d[:,0]>0
        n = self.d.shape[1]
        for i in range(n):
            if i in drug_axes:
                continue
            if other_drug_slices is None:
                dslice = np.min(self.d[:,i])
            else:
                dslice = other_drug_slices[i]
            mask = mask & (self.d[:,i]==dslice)

        d1 = self.d[mask,drug_axes[0]]
        d2 = self.d[mask,drug_axes[1]]
        d3 = self.d[mask,drug_axes[2]]

        Z = self.synergy[mask]
        if neglog:
            Z = -np.log10(Z)

        plots.plotly_isosurfaces(d1, d2, d3, Z, cmap=cmap, center_on_zero=True, **kwargs)
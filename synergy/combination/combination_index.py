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

from synergy.single import Hill_CI
from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.combination.schindler import Schindler


class CombinationIndex(DoseDependentSynergyModel2D):
    """Combination Index (CI).

    CI is a mass-action-derived model of drug combination synergy. In the original 1984 paper (TODO: doi), CI was
    derived with two forms:
        (1) mutually-exclusive drugs, and
        (2) mutually non-exclustive drugs.
    Since then, model (1) has been preferred, and is the one implemented here.

    CI fits single-drug responses using a log-linearization scatchard-like regression, that implicitly assumes E0=1 and
    Emax=0, so should only be applied in those cases. If this assumption is not met, users may wish to use Loewe(),
    which is here calculated identically to CI, but without the E0 and Emax assumptions, and fits drugs to a 4-parameter
    Hill equation using nonlinear optimization.

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
    """

    def E_reference(self, d1, d2):
        """Use the Schindler model to calculate a reference response."""
        reference_model = Schindler(drug1_model=self.drug1_model, drug2_model=self.drug2_model)
        return reference_model.E_reference(d1, d2)

    def _get_synergy(self, d1, d2, E):
        """Calculate CI.

        CI = d1 / E_inv1(E) + d2 / E_inv2(E)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            d1_alone = self.drug1_model.E_inv(E)
            d2_alone = self.drug2_model.E_inv(E)
            synergy = d1 / d1_alone + d2 / d2_alone

        return self._sanitize_synergy(d1, d2, synergy, 1.0)

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """The default drug model to use"""
        return Hill_CI

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """The required superclass of the models for the individual drugs, or None if any model is acceptable"""
        return Hill_CI

    def plot_heatmap(self, cmap="PRGn", neglog=True, center_on_zero=True, **kwargs):
        """-"""
        # super().plot_heatmap(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", neglog=True, center_on_zero=True, **kwargs):
        """-"""
        # super().plot_surface_plotly(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)

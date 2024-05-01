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

from synergy.higher.loewe import Loewe
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill_CI


class CombinationIndex(Loewe):
    """Combination Index (CI).

    CI is a mass-action-derived model of drug combination synergy. In the original 1984 paper (XXX), CI was derived with two forms: (1) mutually-exclusive drugs, and (2) mutually non-exclustive drugs. Since then, model (1) has been preferred, and is the one implemented here.

    CI fits single-drug responses using a log-linearization scatchard-like regression, that implicitly assumes E0=1 and Emax=0, so should only be applied in those cases. If this assumption is not met, users may wish to use Loewe(), which is here calculated identically to CI, but without the E0 and Emax assumptions, and fits drugs to a 4-parameter Hill equation using nonlinear optimization.

    synergy : array_like, float
        [0,1)=synergism, (1,inf)=antagonism
    """

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return Hill_CI

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        return Hill_CI

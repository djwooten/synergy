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

from typing import Type

from synergy.higher.loewe import Loewe
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill_CI


class CombinationIndex(Loewe):
    """The Combination Index (CI) synergy model."""

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill_CI

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill_CI

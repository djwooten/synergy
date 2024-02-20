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

from synergy.combination.nonparametric_base import DoseDependentModel


class Bliss(DoseDependentModel):
    """Bliss independence model

    Bliss synergy is defined as the difference between the observed E and the E predicted by the Bliss Independence assumption (E_pred = E_drug1_alone * E_drug2_alone).

    synergy : array_like, float
        (-inf,0)=antagonism, (0,inf)=synergism
    """

    def _E_reference(self, d1, d2):
        E1_alone = self.drug1_model.E(d1)
        E2_alone = self.drug2_model.E(d2)

        return E1_alone * E2_alone

    def _get_synergy(self, d1, d2, E):
        synergy = self.reference - E
        return self._sanitize_synergy(d1, d2, synergy, 0)

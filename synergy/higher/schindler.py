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

import numpy as np

from synergy.exceptions import InvalidDrugModelError
from synergy.higher.synergy_model_Nd import DoseDependentSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill


class Schindler(DoseDependentSynergyModelND):
    """Schindler's multidimensional Hill equation model."""

    def E_reference(self, d):
        if not self.is_specified:
            raise InvalidDrugModelError("Model is not specified.")

        E0 = 0
        for single in self.single_drug_models:
            E0 += single.E0 / self.N

        with np.errstate(divide="ignore", invalid="ignore"):
            # Schindler assumes drugs start at 0 and go up to Emax
            uE_model = self._model(d, E0)

        uE_model[np.where(d.sum(axis=1) == 0)] = 0  # shindler(d=0) is nan, but we know is really 0
        return E0 - uE_model

    def _get_synergy(self, d, E):
        # E0 = 0
        # for single in self.single_drug_models:
        #    E0 += single.E0 / self.N
        # uE = E0 - E
        # return self._sanitize_synergy(d, uE - self.reference, 0)
        synergy = self.reference - E
        return self._sanitize_synergy(d, synergy, 0)

    def _model(self, d, E0):
        """The synergy model.

        E - u_hill = 0 : Additive
        E - u_hill > 0 : Synergistic
        E - u_hill < 0 : Antagonistic
        """
        h = np.asarray([model.h for model in self.single_drug_models])  # len == N
        C = np.asarray([model.C for model in self.single_drug_models])  # len == N
        Emax = E0 - np.asarray([model.Emax for model in self.single_drug_models])  # len == N

        m = d / C  # shape == (n_points, N)
        msum = m.sum(axis=1)  # len == n_points

        y = (h * m).sum(axis=1) / msum  # len == n_points
        u_max = (Emax * m).sum(axis=1) / msum  # len == n_points
        power = np.float_power(msum, y)  # len == n_points

        return u_max * power / (1.0 + power)

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill

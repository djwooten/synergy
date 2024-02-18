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

import sys

import numpy as np

from synergy.combination import MuSyC
from synergy.higher import MuSyC as MuSyC_higher
from synergy.utils import base as utils, dose_utils
from synergy.single import Hill


FLOAT_MAX = sys.float_info.max


class ShamDataGenerator:
    """Tools to simulate sham combination datasets.

    In a sham experiment, the two drugs combined are (secretly) the same drug. For example, a sham combination may add
    10mg drugA + 20mg drugB. But because drugA and drugB are the same (drugX), the "combination" is equivalent to 30mg
    of the drug.
    """

    @staticmethod
    def get_sham(drug_model, dmin: float, dmax: float, n_points: int, replicates: int = 1, noise: float = 0.05):
        """Return dose and effect data corresponding to a two-drug sham combination experiment."""
        d1, d2 = dose_utils.make_dose_grid(
            dmin, dmax, dmin, dmax, n_points, n_points, replicates=replicates, logscale=False
        )
        E = drug_model.E(d1 + d2)

        if noise > 0:
            E0 = drug_model.E(0)
            Emax = drug_model.E(FLOAT_MAX / 1e50)  # For most cases, this is probably safe to get Emax without overflows
            noise = noise * (E0 - Emax)
            E = E + noise * (2 * np.random.rand(n_points) - 1)

        return d1, d2, E

    @staticmethod
    def get_sham_higher(
        drug_model, dmin: float, dmax: float, n_points: int, n_drugs: int, replicates: int = 1, noise: float = 0.05
    ):
        """Return dose and effect data corresponding to a 3+ drug sham combination experiment."""
        doses = dose_utils.make_dose_grid_multi(dmin, dmax, n_points, logscale=False)
        doses_list = list(np.meshgrid(*doses))
        for i in range(n_drugs):
            doses_list[i] = doses_list[i].flatten()
        doses_array = np.asarray(doses_list).T
        E = drug_model.E(doses_array.sum(axis=1))

        if noise > 0:
            E0 = drug_model.E(0)
            Emax = drug_model.E(FLOAT_MAX / 1e50)  # For most cases, this is probably safe to get Emax without overflows
            noise = noise * (E0 - Emax)
            E = E + noise * (2 * np.random.rand(n_points) - 1)

        return doses, E


class MuSyCDataGenerator:
    """Data generator using the MuSyC model.

    MuSyC can be used to simulate Bliss or Loewe (linear isobole) dose response surfaces.
    """

    def get_2drug_combination(
        self,
        E0: float = 1.0,
        E1: float = 0.5,
        E2: float = 0.3,
        E3: float = 0.0,
        h1: float = 1.0,
        h2: float = 1.0,
        C1: float = 1.0,
        C2: float = 1.0,
        alpha12: float = 1.0,
        alpha21: float = 1.0,
        gamma12: float = 1.0,
        gamma21: float = 1.0,
        noise: float = 0.05,
    ):
        model = MuSyC(
            E0=E0,
            E1=E1,
            E2=E2,
            E3=E3,
            h1=h1,
            h2=h2,
            C1=C1,
            C2=C2,
            alpha12=alpha12,
            alpha21=alpha21,
            gamma12=gamma12,
            gamma21=gamma21,
        )

        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, 6, 6, include_zero=True)

        E = model.E(d1, d2)
        noise = noise * (E0 - E3)

        E = E + noise * (2 * np.random.rand(len(d1)) - 1)

        return d1, d2, E

    def get_2drug_bliss(
        self,
        E1: float = 0.5,
        E2: float = 0.3,
        C1: float = 1.0,
        C2: float = 1.0,
        h1: float = 1.0,
        h2: float = 1.0,
        noise: float = 0.05,
    ):
        if E1 < 0 or E1 > 1 or E2 < 0 or E2 > 1:
            raise ValueError("E1 and E2 must be between 0 and 1 for a Bliss model")
        return self.get_2drug_combination(E0=1.0, E1=E1, E2=E2, E3=E1 * E2, h1=h1, h2=h2, C1=C1, C2=C2, noise=noise)

    def get_2drug_linear_isoboles(
        self,
        E0: float = 1.0,
        E1: float = 0.5,
        E2: float = 0.3,
        C1: float = 1.0,
        C2: float = 1.0,
        h1: float = 1.0,
        h2: float = 1.0,
        noise: float = 0.05,
    ):
        return self.get_2drug_combination(E0=E0, E1=E1, E2=E2, E3=min(E1, E2), h1=h1, h2=h2, C1=C1, C2=C2, noise=noise)

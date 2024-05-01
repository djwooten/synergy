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
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from synergy.combination.bliss import Bliss as Bliss2D
from synergy.higher.bliss import Bliss as BlissND
from synergy.combination.hsa import HSA as Hsa2D
from synergy.higher.hsa import HSA as HsaND
from synergy.combination import MuSyC
from synergy.combination.schindler import Schindler as Schindler2D
from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.combination.braid import BRAID
from synergy.combination.zimmer import Zimmer
from synergy.higher.synergy_model_Nd import DoseDependentSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.utils import dose_utils
from synergy.single import Hill


FLOAT_MAX = sys.float_info.max


def _noisify(vals: np.ndarray, noise: float, min_val: float = np.nan, max_val: float = np.nan) -> np.ndarray:
    """Add relative noise."""
    vals = vals + norm.rvs(scale=vals * noise)
    if not np.isnan(min_val):
        vals[vals < min_val] = min_val
    if not np.isnan(max_val):
        vals[vals > max_val] = max_val
    return vals


class HillDataGenerator:
    """Tools to simulate data from a Hill equation."""

    @staticmethod
    def get_data(
        E0: float = 1.0,
        Emax: float = 0.0,
        h: float = 1.0,
        C: float = 1.0,
        dmin: float = np.nan,
        dmax: float = np.nan,
        n_points: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        if replicates < 1:
            raise ValueError(f"Must have at least 1 replicate ({replicates}).")
        if np.isnan(dmin):
            dmin = C / 20.0
        if np.isnan(dmax):
            dmax = C * 20.0

        if dmin >= dmax:
            raise ValueError(f"dmin ({dmin}) must be less than dmax ({dmax}).")
        if dmin < 0:
            raise ValueError(f"dmin ({dmin}) must be >= 0.")
        if dmin == 0:
            dmin = np.nextafter(0.0)

        d = np.logspace(np.log10(dmin), np.log10(dmax), num=n_points)
        d = np.hstack([d] * replicates)

        d_noisy = _noisify(d, d_noise, min_val=0)

        model = Hill(E0=E0, Emax=Emax, h=h, C=C)
        E = _noisify(model.E(d_noisy), E_noise)
        return d, E


class ShamDataGenerator:
    """Tools to simulate sham combination datasets.

    In a sham experiment, the two drugs combined are (secretly) the same drug. For example, a sham combination may add
    10mg drugA + 20mg drugB. But because drugA and drugB are the same (drugX), the "combination" is equivalent to 30mg
    of the drug.
    """

    @staticmethod
    def get_sham(
        model,
        dmin: float,
        dmax: float,
        n_points: int,
        replicates: int = 1,
        logscale: bool = False,
        include_zero: bool = True,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        """Return dose and effect data corresponding to a two-drug sham combination experiment."""
        d1, d2 = dose_utils.make_dose_grid(
            dmin,
            dmax,
            dmin,
            dmax,
            n_points,
            n_points,
            replicates=replicates,
            logscale=logscale,
            include_zero=include_zero,
        )

        d_noisy = _noisify(d1 + d2, d_noise, min_val=0)
        E = _noisify(model.E(d_noisy), E_noise)

        return d1, d2, E

    @staticmethod
    def get_sham_higher(
        drug_model,
        dmin: float,
        dmax: float,
        n_points: int,
        n_drugs: int,
        replicates: int = 1,
        noise: float = 0.05,
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
            E = E + noise * (2 * np.random.rand(len(E)) - 1)

        return doses, E


class DoseDependentReferenceDataGenerator:
    """-"""

    MODEL: Optional[type[DoseDependentSynergyModel2D]] = None
    MODEL_ND: Optional[type[DoseDependentSynergyModelND]] = None
    MIN_E: float = np.nan
    MAX_E: float = np.nan

    @classmethod
    def get_combination(
        cls,
        drug1_model,
        drug2_model,
        d1min: float,
        d1max: float,
        d2min: float,
        d2max: float,
        n_points1: int = 6,
        n_points2: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        """-"""
        if cls.MODEL is None:
            raise ValueError("No 2-drug model defined for this reference data generator")
        d1, d2 = dose_utils.make_dose_grid(d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=replicates)
        d1_noisy = _noisify(d1, d_noise, min_val=0)
        d2_noisy = _noisify(d2, d_noise, min_val=0)

        model = cls.MODEL(drug1_model, drug2_model)
        E = model.E_reference(d1_noisy, d2_noisy)
        E = _noisify(E, E_noise, min_val=cls.MIN_E, max_val=cls.MAX_E)
        return d1, d2, E

    @classmethod
    def get_ND_combination(
        cls,
        drug_models: Sequence[DoseResponseModel1D],
        dmin: list[float],
        dmax: list[float],
        n_points: list[int],
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        """-"""
        if cls.MODEL_ND is None:
            raise ValueError("No N-drug model defined for this reference data generator")
        d = dose_utils.make_dose_grid_multi(dmin, dmax, n_points, replicates=replicates, include_zero=True)
        d = _noisify(d, d_noise, min_val=0)

        model = cls.MODEL_ND(single_drug_models=drug_models)
        E = model.E_reference(d)
        E = _noisify(E, E_noise, min_val=cls.MIN_E, max_val=cls.MAX_E)
        return d, E


class MultiplicativeSurvivalReferenceDataGenerator(DoseDependentReferenceDataGenerator):
    """-"""

    MODEL: type[DoseDependentSynergyModel2D] = Bliss2D
    MODEL_ND: type[DoseDependentSynergyModelND] = BlissND
    MIN_E = 0
    MAX_E = 1


class HSAReferenceDataGenerator(DoseDependentReferenceDataGenerator):
    """-"""

    MODEL: type[DoseDependentSynergyModel2D] = Hsa2D
    MODEL_ND: type[DoseDependentSynergyModelND] = HsaND


class SchindlerReferenceDataGenerator(DoseDependentReferenceDataGenerator):
    """-"""

    MODEL: type[DoseDependentSynergyModel2D] = Schindler2D
    MODEL_ND: type[DoseDependentSynergyModelND] = None


class MuSyCDataGenerator:
    """Data generator using the MuSyC model.

    MuSyC can be used to simulate Bliss or Loewe (linear isobole) dose response surfaces.
    """

    @staticmethod
    def get_2drug_combination(
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
        d1min=None,
        d1max=None,
        d2min=None,
        d2max=None,
        n_points1: int = 6,
        n_points2: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
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

        if d1min is None:
            d1min = C1 / 20.0
        if d1max is None:
            d1max = C1 * 20.0
        if d2min is None:
            d2min = C2 / 20.0
        if d2max is None:
            d2max = C2 * 20.0

        d1, d2 = dose_utils.make_dose_grid(
            d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=replicates, include_zero=True
        )

        d1_noisy = _noisify(d1, d_noise, min_val=0)
        d2_noisy = _noisify(d2, d_noise, min_val=0)
        E = _noisify(model.E(d1_noisy, d2_noisy), E_noise)
        return d1, d2, E

    @staticmethod
    def get_2drug_bliss(
        E1: float = 0.5,
        E2: float = 0.3,
        C1: float = 1.0,
        C2: float = 1.0,
        h1: float = 1.0,
        h2: float = 1.0,
        d1min=None,
        d1max=None,
        d2min=None,
        d2max=None,
        n_points1: int = 6,
        n_points2: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        if E1 < 0 or E1 > 1 or E2 < 0 or E2 > 1:
            raise ValueError("E1 and E2 must be between 0 and 1 for a Bliss model")
        return MuSyCDataGenerator.get_2drug_combination(
            E0=1.0,
            E1=E1,
            E2=E2,
            E3=E1 * E2,
            h1=h1,
            h2=h2,
            C1=C1,
            C2=C2,
            d1min=d1min,
            d1max=d1max,
            d2min=d2min,
            d2max=d2max,
            n_points1=n_points1,
            n_points2=n_points2,
            replicates=replicates,
            E_noise=E_noise,
            d_noise=d_noise,
        )

    @staticmethod
    def get_2drug_linear_isoboles(
        E0: float = 1.0,
        E1: float = 0.5,
        E2: float = 0.3,
        C1: float = 1.0,
        C2: float = 1.0,
        d1min=None,
        d1max=None,
        d2min=None,
        d2max=None,
        n_points1: int = 6,
        n_points2: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        return MuSyCDataGenerator.get_2drug_combination(
            E0=E0,
            E1=E1,
            E2=E2,
            h1=1,
            h2=1,
            C1=C1,
            C2=C2,
            alpha12=0,
            alpha21=0,
            d1min=d1min,
            d1max=d1max,
            d2min=d2min,
            d2max=d2max,
            n_points1=n_points1,
            n_points2=n_points2,
            replicates=replicates,
            E_noise=E_noise,
            d_noise=d_noise,
        )


class EffectiveDoseModelDataGenerator:
    """Data generator using the effective dose model."""

    @staticmethod
    def get_2drug_combination(
        h1: float = 1.0,
        h2: float = 1.0,
        C1: float = 1.0,
        C2: float = 1.0,
        a12: float = 0.0,
        a21: float = 0.0,
        d1min=None,
        d1max=None,
        d2min=None,
        d2max=None,
        n_points1: int = 6,
        n_points2: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        model = Zimmer(h1=h1, h2=h2, C1=C1, C2=C2, a12=a12, a21=a21)

        if d1min is None:
            d1min = C1 / 20.0
        if d1max is None:
            d1max = C1 * 20.0
        if d2min is None:
            d2min = C2 / 20.0
        if d2max is None:
            d2max = C2 * 20.0

        d1, d2 = dose_utils.make_dose_grid(
            d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=replicates, include_zero=True
        )

        d1_noisy = _noisify(d1, d_noise, min_val=0)
        d2_noisy = _noisify(d2, d_noise, min_val=0)
        E = _noisify(model.E(d1_noisy, d2_noisy), E_noise)
        return d1, d2, E


class BraidDataGenerator:
    """Data generator using the BRAID model."""

    @staticmethod
    def get_2drug_combination(
        E0: float = 1.0,
        E1: float = 0.5,
        E2: float = 0.0,
        E3: float = 0.0,
        h1: float = 1.0,
        h2: float = 1.0,
        C1: float = 1.0,
        C2: float = 1.0,
        delta: float = 1.0,
        kappa: float = 0.0,
        d1min=None,
        d1max=None,
        d2min=None,
        d2max=None,
        n_points1: int = 6,
        n_points2: int = 6,
        replicates: int = 1,
        E_noise: float = 0.05,
        d_noise: float = 0.05,
    ):
        model = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, delta=delta, kappa=kappa, mode="both")

        if d1min is None:
            d1min = C1 / 20.0
        if d1max is None:
            d1max = C1 * 20.0
        if d2min is None:
            d2min = C2 / 20.0
        if d2max is None:
            d2max = C2 * 20.0

        d1, d2 = dose_utils.make_dose_grid(
            d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=replicates, include_zero=False
        )

        d1_noisy = _noisify(d1, d_noise, min_val=0)
        d2_noisy = _noisify(d2, d_noise, min_val=0)
        E = _noisify(model.E(d1_noisy, d2_noisy), E_noise)
        return d1, d2, E

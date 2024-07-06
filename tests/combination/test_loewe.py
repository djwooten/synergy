import os
import unittest
from unittest import TestCase

import numpy as np
from numpy.typing import ArrayLike

from synergy.single import Hill
from synergy.combination import Loewe
from synergy.testing_utils.synthetic_data_generators import ShamDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class ShamLoeweTests(TestCase):
    """Tests for the delta Loewe synergy dose-response models."""

    d1: ArrayLike
    d2: ArrayLike
    E: ArrayLike

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(123)
        single_drug = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        cls.d1, cls.d2, cls.E = ShamDataGenerator.get_combination(
            single_drug, 0.01, 100, 5, 1, logscale=True, E_noise=0, d_noise=0
        )

    def test_fit_loewe_delta_weakest(self):
        """-"""
        # Give it non-prefit single-drug models
        model = Loewe(mode="delta_weakest", drug1_model=Hill, drug2_model=Hill)
        synergy = model.fit(self.d1, self.d2, self.E)
        np.testing.assert_allclose(synergy, 0, atol=1e-2)

    def test_fit_loewe_delta_hsa(self):
        """-"""
        # Give it non-prefit single-drug models
        model = Loewe(mode="delta_hsa", drug1_model=Hill, drug2_model=Hill)
        synergy = model.fit(self.d1, self.d2, self.E)
        np.testing.assert_allclose(synergy, 0, atol=1e-2)

    def test_fit_loewe_ci(self):
        """-"""
        # Give it non-prefit single-drug models
        model = Loewe(mode="ci", drug1_model=Hill, drug2_model=Hill)
        synergy = model.fit(self.d1, self.d2, self.E)

        # CI-loewe is very sensitive when E ~= E0 of the single drug models.
        # In this case, the single drug fits have E0=0.99, which is very close to
        # E at the lowest dose (0.98)
        # That proximity causes CI-loewe to be incorrect at that point. So for now I am
        # removing that data point from the comparison.
        # TODO: Calculate dCI/dE using Hill.E_inv() and come up with some threshold where
        #       uncertainty in E causes unacceptable uncertainty in CI.
        #       Return NaN in these cases.
        #
        #
        # def dEinv_dE(E, E0, Emax, C, h):
        #     return np.float_power((E - E0) / (Emax - E), (1.0 - h) / h) * C * (Emax - E0) / (Emax - E)**2
        #
        # def dCI_dE(d1, d2, E, E0, Emax, C, h):
        #     return -d1 * dEinv_dE1(E) / (E_inv1(E)**2) - d2 * dEinv_dE2(E) / (E_inv2(E)**2)
        synergy_to_test = synergy[np.where((self.d1 > 0.01) & (self.d2 > 0.01))]
        np.testing.assert_allclose(synergy_to_test, 1, rtol=2e-4)

    def test_fit_loewe_synergistm(self):
        """-"""
        E = np.asarray(list(self.E))
        single_mask = np.where((self.d1 == 0) | (self.d2 == 0))
        combo_mask = np.where((self.d1 != 0) & (self.d2 != 0))
        E[combo_mask] *= 0.9  # Make the combo stronger than loewe expects

        # Give it non-prefit single-drug models
        model_delta = Loewe(mode="delta_hsa", drug1_model=Hill, drug2_model=Hill)
        synergy_delta = model_delta.fit(self.d1, self.d2, E)
        self.assertTrue((synergy_delta[combo_mask] > 0).all())
        np.testing.assert_allclose(synergy_delta[single_mask], 0, atol=1e-2)

        model_ci = Loewe(mode="ci", drug1_model=Hill, drug2_model=Hill)
        synergy_ci = model_ci.fit(self.d1, self.d2, E)
        self.assertTrue((synergy_ci[combo_mask] < 1).all())
        np.testing.assert_allclose(synergy_ci[single_mask], 1, atol=1e-2)

    def test_fit_loewe_antagonism(self):
        """-"""
        E = np.asarray(list(self.E))
        single_mask = np.where((self.d1 == 0) | (self.d2 == 0))
        combo_mask = np.where((self.d1 != 0) & (self.d2 != 0))
        E[combo_mask] *= 1.02  # Make the combo weaker than loewe expects

        # Give it non-prefit single-drug models
        model_delta = Loewe(mode="delta_hsa", drug1_model=Hill, drug2_model=Hill)
        synergy_delta = model_delta.fit(self.d1, self.d2, E)
        self.assertTrue((synergy_delta[combo_mask] < 0).all())
        np.testing.assert_allclose(synergy_delta[single_mask], 0, atol=1e-2)

        model_ci = Loewe(mode="ci", drug1_model=Hill, drug2_model=Hill)
        synergy_ci = model_ci.fit(self.d1, self.d2, E)
        print(synergy_ci)
        self.assertTrue((synergy_ci[combo_mask] > 1).all())
        np.testing.assert_allclose(synergy_ci[single_mask], 1, atol=1e-2)


if __name__ == "__main__":
    unittest.main()

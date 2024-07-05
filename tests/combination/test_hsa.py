import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single import Hill
from synergy.combination import HSA
from synergy.testing_utils.synthetic_data_generators import HSAReferenceDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class HSATests(TestCase):
    """Tests for the HSA model."""

    def test_fit_reference(self):
        """-"""
        np.random.seed(2193)
        dmin = 0.01
        dmax = 100
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        d1, d2, E = HSAReferenceDataGenerator.get_combination(
            drug1, drug2, dmin, dmax, dmin, dmax, E_noise=0, d_noise=0
        )

        model = HSA()
        synergy = model.fit(d1, d2, E)

        np.testing.assert_almost_equal(synergy, 0)

    def test_fit_synergy(self):
        """Ensure a synergistic combination has synergy > 0 when d1 and d2 > 0, and 0 when d1 or d2 == 0"""
        np.random.seed(81924)
        dmin = 0.01
        dmax = 100
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        d1, d2, E = HSAReferenceDataGenerator.get_combination(
            drug1, drug2, dmin, dmax, dmin, dmax, E_noise=0, d_noise=0, include_zero=True
        )

        single_mask = np.where((d1 == 0) | (d2 == 0))
        combo_mask = np.where((d1 != 0) & (d2 != 0))
        E[combo_mask] *= 0.9  # Make the combo stronger than HSA expects

        model = HSA()
        synergy = model.fit(d1, d2, E)

        self.assertTrue((synergy[combo_mask] > 0).all())
        np.testing.assert_almost_equal(synergy[single_mask], 0)

    def test_fit_antagonism(self):
        """Ensure an antagonistic combination has synergy < 0 when d1 and d2 > 0, and 0 when d1 or d2 == 0"""
        np.random.seed(891248)
        dmin = 0.01
        dmax = 100
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        d1, d2, E = HSAReferenceDataGenerator.get_combination(
            drug1, drug2, dmin, dmax, dmin, dmax, E_noise=0, d_noise=0, include_zero=True
        )

        single_mask = np.where((d1 == 0) | (d2 == 0))
        combo_mask = np.where((d1 != 0) & (d2 != 0))
        E[combo_mask] *= 1.1  # Make the combo weaker than HSA expects

        model = HSA()
        synergy = model.fit(d1, d2, E)

        self.assertTrue((synergy[combo_mask] < 0).all())
        np.testing.assert_almost_equal(synergy[single_mask], 0)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single import Hill
from synergy.combination import ZIP
from synergy.testing_utils.synthetic_data_generators import (
    MultiplicativeSurvivalReferenceDataGenerator,
    MuSyCDataGenerator,
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class ZIPTests(TestCase):
    """Tests for the ZIP model."""

    def test_fit_zip_reference(self):
        """Ensure ZIP has ~0 synergy for a synthetic Bliss-Independent combination."""
        np.random.seed(943)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        d1, d2, E = MultiplicativeSurvivalReferenceDataGenerator.get_combination(
            drug1, drug2, 0.01, 100, 0.01, 100, 5, 5, E_noise=0, d_noise=0
        )

        model = ZIP()
        synergy = model.fit(d1, d2, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)), atol=2e-2)  # TODO it seems like atol is high...

    def test_fit_zip_synergy(self):
        """Ensure a synergistic combination has synergy > 0 when d1 and d2 > 0, and 0 when d1 or d2 == 0"""
        np.random.seed(81924)
        # Use MuSyC to simulate synergistic potency on an otherwise Bliss-Independent drug pair
        d1, d2, E = MuSyCDataGenerator.get_2drug_combination(
            E0=1, E1=0.5, E2=0.3, E3=0.15, alpha12=2, alpha21=2, E_noise=0, d_noise=0
        )

        model = ZIP()
        synergy = model.fit(d1, d2, E)
        single_mask = np.where((d1 == 0) | (d2 == 0))
        combo_mask = np.where((d1 != 0) & (d2 != 0))
        self.assertTrue((synergy[combo_mask] > 0).all())
        np.testing.assert_almost_equal(synergy[single_mask], 0)

    def test_fit_zip_antagonism(self):
        """Ensure an antagonistic combination has synergy < 0 when d1 and d2 > 0, and 0 when d1 or d2 == 0"""
        np.random.seed(891248)
        # Use MuSyC to simulate antagonistic potency on an otherwise Bliss-Independent drug pair
        d1, d2, E = MuSyCDataGenerator.get_2drug_combination(
            E0=1, E1=0.5, E2=0.3, E3=0.15, alpha12=0.5, alpha21=0.5, E_noise=0, d_noise=0
        )

        model = ZIP()
        synergy = model.fit(d1, d2, E)
        single_mask = np.where((d1 == 0) | (d2 == 0))
        combo_mask = np.where((d1 != 0) & (d2 != 0))
        self.assertTrue((synergy[combo_mask] < 0).all())
        np.testing.assert_almost_equal(synergy[single_mask], 0)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single.hill import Hill
from synergy.combination import Loewe
from synergy.testing_utils.synthetic_data_generators import ShamDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class DeltaLoeweTests(TestCase):
    """Tests for the delta Loewe synergy dose-response models."""

    def test_fit_loewe(self):
        """-"""
        np.random.seed(123)
        single_drug = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        d1, d2, E = ShamDataGenerator.get_sham(single_drug, 0.01, 100, 5, 2, E_noise=0, d_noise=0)

        # Give it non-prefit single-drug models
        model = Loewe(mode="delta", drug1_model=Hill, drug2_model=Hill)
        synergy = model.fit(d1, d2, E)
        np.testing.assert_allclose(synergy, synergy * 0.0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()

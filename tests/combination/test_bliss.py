import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single.hill import Hill
from synergy.combination import Bliss
from synergy.testing_utils.synthetic_data_generators import ShamDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class HSATests(TestCase):
    """Tests for the HSA model."""

    def test_fit_hsa(self):
        """-"""
        np.random.seed(943)
        single_drug = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        d1, d2, E = ShamDataGenerator.get_sham(single_drug, 0.01, 100, 5, 2, E_noise=0, d_noise=0)
        E1 = single_drug.E(d1)
        E2 = single_drug.E(d2)
        E = E1 * E2

        # Give it non-prefit single-drug models
        model = Bliss()
        synergy = model.fit(d1, d2, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)), atol=2e-2)  # TODO this should be closer than this


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single.hill import Hill
from synergy.combination import HSA
from synergy.testing_utils.synthetic_data_generators import ShamDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class HSATests(TestCase):
    """Tests for the HSA model."""

    def test_fit_hsa(self):
        """-"""
        np.random.seed(2193)
        single_drug = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        d1, d2, E = ShamDataGenerator.get_sham(single_drug, 0.01, 100, 5, 2, E_noise=0, d_noise=0)

        # Give it non-prefit single-drug models
        model = HSA()
        synergy = model.fit(d1, d2, E)
        # synergy should be > 0 for all of these
        self.assertTrue((synergy >= 0).all(), msg="HSA should all be synergistic")


if __name__ == "__main__":
    unittest.main()

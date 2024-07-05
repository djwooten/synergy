import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single.hill import Hill
from synergy.combination import Schindler
from synergy.testing_utils.synthetic_data_generators import ShamDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class SchindlerTests(TestCase):
    """Tests for the Schindler synergy model."""

    def test_fit_sham(self):
        """-"""
        np.random.seed(49)
        single_drug = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        d1, d2, E = ShamDataGenerator.get_combination(single_drug, 0.01, 100, 5, 2, E_noise=0, d_noise=0)

        model = Schindler()
        synergy = model.fit(d1, d2, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)), atol=1e-2)


if __name__ == "__main__":
    unittest.main()

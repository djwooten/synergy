import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.combination import CombinationIndex
from synergy.single import Hill_CI
from synergy.testing_utils.synthetic_data_generators import ShamDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class CombinationIndexTests(TestCase):
    """Tests for the combination index synergy model."""

    def test_fit_ci(self):
        """-"""
        np.random.seed(89234)
        single_drug = Hill_CI(h=1.0, C=1.0)
        d1, d2, E = ShamDataGenerator.get_combination(single_drug, 0.01, 10, 6, 2, E_noise=0, d_noise=0)

        model = CombinationIndex(drug1_model=single_drug, drug2_model=single_drug)
        synergy = model.fit(d1, d2, E)
        np.testing.assert_allclose(synergy, np.ones(len(synergy)), atol=1e-2)


if __name__ == "__main__":
    unittest.main()

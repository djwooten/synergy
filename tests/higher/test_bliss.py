import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single.hill import Hill
from synergy.higher import Bliss
from synergy.testing_utils.synthetic_data_generators import BlissDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class BlissTests(TestCase):
    """Tests for the Bliss Independence model."""

    def test_fit_bliss(self):
        """-"""
        np.random.seed(943)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        d1, d2, E = BlissDataGenerator.get_2drug_combination(
            drug1, drug2, 0.01, 100, 0.01, 100, 5, 5, E_noise=0, d_noise=0
        )

        # Give it non-prefit single-drug models
        model = Bliss()
        synergy = model.fit(d1, d2, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)), atol=2e-2)  # TODO it seems like atol is high...


if __name__ == "__main__":
    unittest.main()

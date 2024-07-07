import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.higher import Bliss
from synergy.single.hill import Hill
from synergy.testing_utils.synthetic_data_generators import (
    MultiplicativeSurvivalReferenceDataGenerator,
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class BlissTests(TestCase):
    """Tests for the Bliss Independence model."""

    def test_fit_bliss(self):
        """-"""
        np.random.seed(943)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.2, h=2.0, C=1.0)

        single_drugs = [drug1, drug2, drug3]
        dmin = [1e-2, 1e-2, 1e-2]
        dmax = [1e2, 1e2, 1e2]
        n_points = [5, 5, 5]

        d, E = MultiplicativeSurvivalReferenceDataGenerator.get_ND_combination(
            single_drugs, dmin, dmax, n_points, E_noise=0, d_noise=0
        )

        # Give it non-prefit single-drug models
        model = Bliss()
        synergy = model.fit(d, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)))


if __name__ == "__main__":
    unittest.main()

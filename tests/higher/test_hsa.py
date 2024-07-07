import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.higher import HSA
from synergy.single.hill import Hill
from synergy.testing_utils.synthetic_data_generators import HSAReferenceDataGenerator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class HSATests(TestCase):
    """Tests for the N-drug HSA model."""

    def test_fit_reference(self):
        """Ensure N-drug HSA model gives 0 synergy for its reference model"""
        np.random.seed(65468)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.2, h=2.0, C=1.0)

        single_drugs = [drug1, drug2, drug3]
        dmin = [1e-2, 1e-2, 1e-2]
        dmax = [1e2, 1e2, 1e2]
        n_points = [5, 5, 5]

        d, E = HSAReferenceDataGenerator.get_ND_combination(single_drugs, dmin, dmax, n_points, E_noise=0, d_noise=0)

        model = HSA()
        synergy = model.fit(d, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)))

    def test_fit_synergy(self):
        """Ensure N-drug HSA model gives synergy > 0 for data stronger than its reference."""
        np.random.seed(43092)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.2, h=2.0, C=1.0)

        single_drugs = [drug1, drug2, drug3]
        dmin = [1e-2, 1e-2, 1e-2]
        dmax = [1e2, 1e2, 1e2]
        n_points = [5, 5, 5]

        d, E = HSAReferenceDataGenerator.get_ND_combination(single_drugs, dmin, dmax, n_points, E_noise=0, d_noise=0)
        mask = np.where(d.sum(axis=1) > 0)
        E[mask] = E[mask] * 0.9  # Make the combo data stronger

        model = HSA()
        synergy = model.fit(d, E)
        # import pdb

        # pdb.set_trace()
        self.assertTrue((synergy >= 0).all())


if __name__ == "__main__":
    unittest.main()

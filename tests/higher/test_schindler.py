import os
import unittest
from unittest import TestCase

import numpy as np

import synergy.testing_utils.synthetic_data_generators as generators
from synergy.higher.schindler import Schindler
from synergy.single.hill import Hill

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class SchindlerNDTests(TestCase):
    """Tests for the Schindler N-drug model."""

    def test_fit_reference(self):
        """Ensure N-drug Schindler model gives 0 synergy for its reference model"""
        np.random.seed(65468)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.2, h=2.0, C=1.0)

        single_drugs = [drug1, drug2, drug3]
        dmin = [1e-2, 1e-2, 1e-2]
        dmax = [1e2, 1e2, 1e2]
        n_points = [5, 5, 5]

        d, E = generators.SchindlerReferenceDataGenerator.get_ND_combination(
            single_drugs, dmin, dmax, n_points, E_noise=0, d_noise=0
        )

        model = Schindler()
        synergy = model.fit(d, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)), atol=1e-15)

    def test_fit_sham(self):
        """Ensure N-drug Schindler model gives 0 synergy for a sham experiment"""
        np.random.seed(743987598)
        hill = Hill(E0=1.0, Emax=0.0, h=2.0, C=1.0)
        d, E = generators.ShamDataGenerator.get_ND_combination(hill, 3, 0.1, 1, E_noise=0, d_noise=0)

        model = Schindler()
        synergy = model.fit(d, E)
        np.testing.assert_allclose(synergy, np.zeros(len(synergy)), atol=1e-15)

    def test_msa_synergy(self):
        """Ensure N-drug Schindler model gives synergy > 0 for Bliss data.

        This should hold true when the drugs have h=1, since when h=1, MuSyC produces
        sham data with alpha=0, whereas it produces Bliss data with alpha=1
        """
        np.random.seed(89174)
        drug1 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.3, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.2, h=1.0, C=1.0)
        d, E = generators.MultiplicativeSurvivalReferenceDataGenerator.get_ND_combination(
            drug_models=[drug1, drug2, drug3], E_noise=0, d_noise=0
        )

        model = Schindler()
        synergy = model.fit(d, E)
        self.assertTrue((synergy >= np.zeros(len(synergy))).all())


if __name__ == "__main__":
    unittest.main()

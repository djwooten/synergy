import os
import unittest
from unittest import TestCase

import numpy as np

import synergy.testing_utils.synthetic_data_generators as generators
from synergy.higher.loewe import Loewe
from synergy.single.hill import Hill
from synergy.utils.dose_utils import is_monotherapy_ND

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class LoeweNDTests(TestCase):
    """Tests for the Loewe N-drug model."""

    def test_fit_reference(self):
        """Ensure N-drug Loewe model gives 0 synergy for its reference model"""
        np.random.seed(894)
        drug1 = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)

        single_drugs = [drug1, drug2, drug3]
        dmin = [1e-2, 1e-2, 1e-2]
        dmax = [1e2, 1e2, 1e2]
        n_points = [10, 10, 10]

        # TODO implement E_reference for loewe so that we can use it (and h != 1) instead of Schindler
        d, E = generators.SchindlerReferenceDataGenerator.get_ND_combination(
            single_drugs, dmin, dmax, n_points, E_noise=0, d_noise=0
        )

        model = Loewe(single_drug_models=[Hill, Hill, Hill])
        synergy = model.fit(d, E)
        log_synergy = np.log(synergy)  # log-scale synergy so it is centered at 0
        np.testing.assert_allclose(log_synergy, np.zeros(len(log_synergy)), atol=1e-13)

    def test_fit_sham(self):
        """Ensure N-drug Loewe model gives 0 synergy for a sham experiment"""
        np.random.seed(684684)
        hill = Hill(E0=1.0, Emax=0.0, h=2.0, C=1.0)
        d, E = generators.ShamDataGenerator.get_ND_combination(hill, 3, 0.1, 1, E_noise=0, d_noise=0)

        # Using log-linear has problems since at the tested dose range, Emax exceeds the single drugs
        model = Loewe(single_drug_models=[Hill, Hill, Hill])
        synergy = model.fit(d, E)
        log_synergy = np.log(synergy)  # log-scale synergy so it is centered at 0
        np.testing.assert_allclose(log_synergy, np.zeros(len(log_synergy)), atol=1e-13)

    def test_msa_synergy(self):
        """Ensure N-drug Loewe model gives synergy > 0 for Bliss data.

        This should hold true when the drugs have h=1, since when h=1, MuSyC produces
        sham data with alpha=0, whereas it produces Bliss data with alpha=1
        """
        np.random.seed(7487777)
        drug1 = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        drug2 = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        drug3 = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)
        d, E = generators.MultiplicativeSurvivalReferenceDataGenerator.get_ND_combination(
            drug_models=[drug1, drug2, drug3], E_noise=0, d_noise=0
        )

        model = Loewe(single_drug_models=[Hill, Hill, Hill])
        synergy = model.fit(d, E)
        combo_indices = np.where(~np.apply_along_axis(is_monotherapy_ND, 1, d))
        self.assertTrue((synergy[combo_indices] < np.ones(len(synergy[combo_indices]))).all())


if __name__ == "__main__":
    unittest.main()

# TODO: Ensure proper behavior of bounds when fitting

import os
import sys
import unittest
from copy import deepcopy
from typing import Dict
from unittest import TestCase

import numpy as np

from synergy.combination import MuSyC
from synergy.testing_utils import assertions as synergy_assertions
from synergy.testing_utils.test_data_loader import load_test_data
from synergy.utils import dose_utils

MAX_FLOAT = sys.float_info.max

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _get_E3(E0, E1, E2, beta):
    strongest_E = np.amin(np.asarray([E1, E2]), axis=0)
    return strongest_E - beta * (E0 - strongest_E)


class MuSyCTests(TestCase):
    """Tests for the 2D MuSyC synergy model."""

    def test_is_specified_and_fit_at_creation(self):
        """Ensure expected behaviors of is_specified and is_fit when creating (but not fitting) models."""
        non_specified_model = MuSyC()

        specified_model_gamma = MuSyC(
            E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, alpha12=1, alpha21=1, gamma12=1, gamma21=1, fit_gamma=True
        )

        specified_model_nogamma = MuSyC(
            E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, alpha12=1, alpha21=1, fit_gamma=False
        )

        nan_model = MuSyC(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, alpha12=1, alpha21=np.nan, fit_gamma=False)

        none_model = MuSyC(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, alpha12=1, alpha21=None, fit_gamma=False)

        for model in [non_specified_model, nan_model, none_model]:
            self.assertFalse(model.is_specified)
            self.assertFalse(model.is_fit)

        for model in [specified_model_gamma, specified_model_nogamma]:
            self.assertTrue(model.is_specified)
            self.assertFalse(model.is_fit)

    def test_no_scores_if_model_specified(self):
        """Ensure the model scores are not present if the model was not fit to data."""
        model = MuSyC(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, alpha12=1, alpha21=1, fit_gamma=False)
        self.assertFalse(hasattr(model, "aic"))
        self.assertFalse(hasattr(model, "bic"))
        self.assertFalse(hasattr(model, "r_squared"))
        self.assertFalse(hasattr(model, "sum_of_squares_residuals"))

    def test_asymptotic_limits(self):
        """Ensure model behaves well when d=0, d->inf, or d=C

        Constraints are placed on values to avoid:
            - Overflows for d**h + C**h
            - Overflows for E0 - Emax
            - Numerical problems (in the test) calculating differences for very large numbers
            - Numerical problems (in the test) calculating ratios of numbers near 0 (< 1e-6)
        """
        E0 = 1
        E1 = 0.4
        E2 = 0.2
        E3 = 0.0
        logh1 = -1
        logh2 = 1
        logC1 = 0
        logC2 = 0
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)

        # This will get d**h + C**h as close to MAX_FLOAT as possible
        d1max = MAX_FLOAT ** (1 / max(h1, 1.0)) - C1**h1
        d2max = MAX_FLOAT ** (1 / max(h2, 1.0)) - C2**h2
        # This helps avoid overflows
        d1max = np.power(d1max, 0.4)
        d2max = np.power(d2max, 0.4)

        model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=1.0, alpha21=1.0, fit_gamma=False)

        # Ensure E(0) == E0
        self.assertEqual(model.E(0, 0), E0, "E(0) should be E0")

        # Ensure E(inf, 0) == E1 and E(0, inf) == E2 and E(inf, inf) == E3
        for E, d in zip([E1, E2, E3], [(d1max, 0), (0, d2max), (d1max, d2max)]):
            d1, d2 = d
            d1s = "0" if d1 == 0 else "inf"
            d2s = "0" if d2 == 0 else "inf"

            if np.abs(E) < 1e-6:  # For tiny values, compare E(inf) == Emax
                self.assertAlmostEqual(model.E(d1, d2), E, places=1, msg=f"E({d1s}, {d2s}) should be {E:0.3g}")
            else:  # For larger values, compare E(inf) / Emax ~= 1
                self.assertAlmostEqual(model.E(d1, d2) / E, 1.0, places=2, msg=f"E({d1s}, {d2s}) should be {E:0.3g}")

        for E, d in zip(
            [(E0 + E1) / 2, (E0 + E2) / 2, (E2 + E3) / 2, (E1 + E3) / 2], [(C1, 0), (0, C2), (C1, d2max), (d1max, C2)]
        ):
            d1, d2 = d
            d1s = "0" if d1 == 0 else ("C1" if d1 == C1 else "inf")
            d2s = "0" if d2 == 0 else ("C2" if d2 == C2 else "inf")
            if np.abs(E) < 1e-6:
                self.assertAlmostEqual(model.E(d1, d2), E, places=1, msg=f"E({d1s}, {d2s}) should be {E:0.3g}")
            else:
                self.assertAlmostEqual(model.E(d1, d2) / E, 1.0, places=2, msg=f"E({d1s}, {d2s}) should be {E:0.3g}")

    def test_synergism(self):
        """Ensure synergistic parameters lead to stronger E"""
        E0, E1, E2, h1, h2, C1, C2 = 1, 0.5, 0.3, 1, 1, 1, 1
        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, n_points1=6, n_points2=6)
        for beta, alpha12, alpha21 in [(0, 2, 1), (0, 1, 2), (0, 2, 2), (1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)]:
            E3 = _get_E3(E0, E1, E2, beta)
            model = MuSyC(
                E0=E0,
                E1=E1,
                E2=E2,
                E3=E3,
                h1=h1,
                h2=h2,
                C1=C1,
                C2=C2,
                alpha12=alpha12,
                alpha21=alpha21,
                gamma12=1,
                gamma21=1,
            )
            E = model.E(d1, d2)
            reference = model.E_reference(d1, d2)
            self.assertTrue((E < reference).all())

    def test_antagonism(self):
        """Ensure antagonistic parameters lead to weaker E"""
        E0, E1, E2, h1, h2, C1, C2 = 1, 0.5, 0.3, 1, 1, 1, 1
        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, n_points1=6, n_points2=6)
        for beta, alpha12, alpha21 in [
            (0, 0.5, 1),
            (0, 1, 0.5),
            (0, 0.5, 0.5),
            (-1, 1, 1),
            (-1, 0.5, 1),
            (-1, 1, 0.5),
            (-1, 0.5, 0.5),
        ]:
            E3 = _get_E3(E0, E1, E2, beta)
            model = MuSyC(
                E0=E0,
                E1=E1,
                E2=E2,
                E3=E3,
                h1=h1,
                h2=h2,
                C1=C1,
                C2=C2,
                alpha12=alpha12,
                alpha21=alpha21,
                gamma12=1,
                gamma21=1,
            )
            E = model.E(d1, d2)
            reference = model.E_reference(d1, d2)
            self.assertTrue((E > reference).all())


class MuSyCFitTests(TestCase):
    """Tests requiring fitting the 2D MuSyC synergy model."""

    EXPECTED_PARAMETERS: Dict[str, Dict[str, float]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.EXPECTED_PARAMETERS = {
            "synthetic_musyc_reference_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0,
                "E3": 0,
                "h1": 1.2,
                "h2": 0.8,
                "C1": 1,
                "C2": 1,
                "alpha12": 1,
                "alpha21": 1,
            },
            "synthetic_musyc_efficacy_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.3,
                "E3": 0,
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "alpha12": 1,
                "alpha21": 1,
            },
            "synthetic_musyc_efficacy_2.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0,
                "E3": 0.3,
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "alpha12": 1,
                "alpha21": 1,
            },
            "synthetic_musyc_potency_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.3,
                "E3": 0,
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "alpha12": 0.5,
                "alpha21": 2,
            },
            "synthetic_musyc_cooperativity_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0,
                "E3": 0,
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "alpha12": 1,
                "alpha21": 1,
                "gamma12": 0.5,
                "gamma21": 2,
            },
        }

    def test_fit_musyc_no_bootstrap(self):
        """Ensure the model fits correctly."""
        for fname in [
            "synthetic_musyc_reference_1.csv",
            "synthetic_musyc_efficacy_1.csv",
            "synthetic_musyc_efficacy_2.csv",
            "synthetic_musyc_potency_1.csv",
        ]:
            np.random.seed(2340214390)
            d1, d2, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
            model = MuSyC(fit_gamma=False)
            model.fit(d1, d2, E)

            # Ensure the hill is fit
            self.assertTrue(model.is_specified)
            self.assertTrue(model.is_fit)
            self.assertTrue(model.is_converged)

            # Compare C, h, alpha, and gamma in log-scale
            expected_parameters = deepcopy(self.EXPECTED_PARAMETERS[fname])
            observed_parameters = model.get_parameters()
            for key in ["alpha12", "alpha21", "gamma12", "gamma21", "h1", "h2", "C1", "C2"]:
                if key in expected_parameters and key in observed_parameters:
                    expected_parameters[key] = np.log(expected_parameters[key])
                    observed_parameters[key] = np.log(observed_parameters[key])

            # Ensure the parameters are approximately correct
            synergy_assertions.assert_dict_allclose(observed_parameters, expected_parameters, atol=0.25, err_msg=fname)

            # Ensure the scores were calculated
            self.assertIsNotNone(model.aic)
            self.assertIsNotNone(model.bic)
            self.assertGreaterEqual(model.r_squared, 0, msg="r_squared should be between 0 and 1")
            self.assertLessEqual(model.r_squared, 1, msg="r_squared should be less than 1")
            self.assertGreaterEqual(model.sum_of_squares_residuals, 0, msg="sum_of_squares_residuals should be >= 0")

            # Ensure there were no bootstrap iterations
            self.assertIsNone(model.bootstrap_parameters)
            with self.assertRaises(ValueError):
                _ = model.get_confidence_intervals()

    def test_musyc_fit_bootstrap(self):
        """Ensure confidence intervals work reasonably."""
        for fname in [
            "synthetic_musyc_reference_1.csv",
            "synthetic_musyc_efficacy_1.csv",
            "synthetic_musyc_efficacy_2.csv",
            "synthetic_musyc_potency_1.csv",
        ]:
            np.random.seed(24309184)
            expected_parameters = deepcopy(self.EXPECTED_PARAMETERS[fname])

            d1, d2, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
            model = MuSyC(fit_gamma=False)
            model.fit(d1, d2, E, bootstrap_iterations=100)

            # Ensure there were bootstrap iterations
            self.assertIsNotNone(model.bootstrap_parameters)

            confidence_intervals_95 = model.get_confidence_intervals()

            # We must add beta, because it is reported in CIs, but not parameters
            expected_parameters["beta"] = MuSyC._get_beta(
                expected_parameters["E0"],
                expected_parameters["E1"],
                expected_parameters["E2"],
                expected_parameters["E3"],
            )

            # Ensure true values are within confidence intervals
            synergy_assertions.assert_dict_values_in_intervals(
                expected_parameters, confidence_intervals_95, err_msg=fname, tol=0.05
            )

            # Ensure that less stringent CI is narrower
            # [=====95=====]  More confidence requires wider interval
            #     [===50==]   Less confidence but tighter interval
            confidence_intervals_50 = model.get_confidence_intervals(confidence_interval=50)
            synergy_assertions.assert_dict_interval_is_contained_in_other(
                confidence_intervals_50, confidence_intervals_95
            )


if __name__ == "__main__":
    unittest.main()

# TODO: Ensure proper behavior of bounds when fitting

import os
import sys
import unittest
from copy import deepcopy
from unittest import TestCase

import hypothesis
import numpy as np
from hypothesis import given
from hypothesis.strategies import sampled_from

from synergy.combination import Zimmer
from synergy.testing_utils.test_data_loader import load_test_data
from synergy.testing_utils import assertions as synergy_assertions
from synergy.utils import dose_utils


MAX_FLOAT = sys.float_info.max

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class EffectiveDoseModelTests(TestCase):
    """Tests for the 2D effective dose synergy model."""

    def test_is_specified_and_fit_at_creation(self):
        """Ensure expected behaviors of is_specified and is_fit when creating (but not fitting) models."""
        non_specified_model = Zimmer()

        specified_model = Zimmer(h1=1, h2=2, C1=1, C2=1, a12=0, a21=0)
        nan_model = Zimmer(h1=1, h2=2, C1=1, C2=1, a12=0, a21=np.nan)
        none_model = Zimmer(h1=1, h2=2, C1=1, C2=1, a12=0)

        for model in [non_specified_model, nan_model, none_model]:
            self.assertFalse(model.is_specified)
            self.assertFalse(model.is_fit)

        for model in [specified_model]:
            self.assertTrue(model.is_specified)
            self.assertFalse(model.is_fit)

    def test_no_scores_if_model_specified(self):
        """Ensure the model scores are not present if the model was not fit to data."""
        model = Zimmer(h1=1, h2=2, C1=1, C2=1, a12=0, a21=0)
        self.assertFalse(hasattr(model, "aic"))
        self.assertFalse(hasattr(model, "bic"))
        self.assertFalse(hasattr(model, "r_squared"))
        self.assertFalse(hasattr(model, "sum_of_squares_residuals"))

    def test_asymptotic_limits(self):
        """Ensure model behaves well when d=0, d->inf, or d=C"""
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
        d1max = np.power(d1max, 0.1)
        d2max = np.power(d2max, 0.1)

        model = Zimmer(h1=h1, h2=h2, C1=C1, C2=C1, a12=0, a21=0)

        # Ensure E(0) == E0
        self.assertEqual(model.E(0, 0), 1, "E(0, 0) should be 1")

        # Ensure E(inf, 0) == E1 and E(0, inf) == E2 and E(inf, inf) == E3
        for d1, d2 in [(d1max, 0), (0, d2max), (d1max, d2max)]:
            d1s = "0" if d1 == 0 else "inf"
            d2s = "0" if d2 == 0 else "inf"

            self.assertAlmostEqual(model.E(d1, d2), 0, places=3, msg=f"E({d1s}, {d2s}) should be 0")

        for E, d in zip([0.5, 0.5, 0, 0], [(C1, 0), (0, C2), (C1, d2max), (d1max, C2)]):
            d1, d2 = d
            d1s = "0" if d1 == 0 else ("C1" if d1 == C1 else "inf")
            d2s = "0" if d2 == 0 else ("C2" if d2 == C2 else "inf")
            self.assertAlmostEqual(model.E(d1, d2), E, places=3, msg=f"E({d1s}, {d2s}) should be {E}")

    def test_synergism(self):
        """Ensure synergistic parameters lead to stronger E"""
        C1, C2 = 1, 1
        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, n_points1=6, n_points2=6)
        for a12, a21 in [(-1, 0), (0, -1), (-0.5, -0.5)]:
            model = Zimmer(h1=1, h2=1, C1=C1, C2=C2, a12=a12, a21=a21)
            E = model.E(d1, d2)
            reference = model.E_reference(d1, d2)
            self.assertTrue((E < reference).all())

    def test_antagonism(self):
        """Ensure antagonistic parameters lead to weaker E"""
        C1, C2 = 1, 1
        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, n_points1=6, n_points2=6)
        for a12, a21 in [(1, 0), (0, 1), (1, 1)]:
            model = Zimmer(h1=1, h2=1, C1=C1, C2=C2, a12=a12, a21=a21)
            E = model.E(d1, d2)
            reference = model.E_reference(d1, d2)
            self.assertTrue((E > reference).all())


class ZimmerFitTests(TestCase):
    """Tests requiring fitting the 2D Zimmer synergy model."""

    EXPECTED_PARAMETERS: dict[str, dict[str, float]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.EXPECTED_PARAMETERS = {
            "synthetic_EDM_reference_1.csv": {
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "a12": 0,
                "a21": 0,
            },
            "synthetic_EDM_synergy_1.csv": {
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "a12": -0.5,
                "a21": 0,
            },
            "synthetic_EDM_synergy_2.csv": {
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "a12": 0,
                "a21": -0.5,
            },
            "synthetic_EDM_synergy_3.csv": {
                "h1": 0.8,
                "h2": 1.4,
                "C1": 1,
                "C2": 1,
                "a12": -0.5,
                "a21": -0.5,
            },
        }

    @given(
        sampled_from(
            [
                "synthetic_EDM_reference_1.csv",
                "synthetic_EDM_synergy_1.csv",
                "synthetic_EDM_synergy_2.csv",
                "synthetic_EDM_synergy_3.csv",
            ]
        )
    )
    def test_fit_no_bootstrap(self, fname):
        """Ensure the model fits correctly."""
        d1, d2, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
        model = Zimmer()
        model.fit(d1, d2, E, use_jacobian=False)

        # Ensure the model is fit
        self.assertTrue(model.is_specified)
        self.assertTrue(model.is_fit)
        self.assertTrue(model.is_converged)

        # Compare C and h in log-scale
        expected_parameters = deepcopy(self.EXPECTED_PARAMETERS[fname])
        observed_parameters = model.get_parameters()
        for key in ["h1", "h2", "C1", "C2"]:
            if key in expected_parameters and key in observed_parameters:
                expected_parameters[key] = np.log(expected_parameters[key])
                observed_parameters[key] = np.log(observed_parameters[key])

        # Ensure the parameters are approximately correct
        synergy_assertions.assert_dict_allclose(observed_parameters, expected_parameters, atol=0.2)

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

    @hypothesis.settings(deadline=None)
    @given(
        sampled_from(
            [
                "synthetic_EDM_reference_1.csv",
                "synthetic_EDM_synergy_1.csv",
                "synthetic_EDM_synergy_2.csv",
                "synthetic_EDM_synergy_3.csv",
            ]
        )
    )
    def test_fit_bootstrap(self, fname):
        """Ensure confidence intervals work reasonably."""
        expected_parameters = deepcopy(self.EXPECTED_PARAMETERS[fname])

        d1, d2, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
        model = Zimmer()
        model.fit(d1, d2, E, use_jacobian=False, bootstrap_iterations=100)

        # Ensure there were bootstrap iterations
        self.assertIsNotNone(model.bootstrap_parameters)

        confidence_intervals_95 = model.get_confidence_intervals()

        log_ci_95 = deepcopy(confidence_intervals_95)
        log_ex_params = deepcopy(expected_parameters)

        for key in ["h1", "h2", "C1", "C2"]:
            log_ci_95["log" + key] = np.log(log_ci_95.pop(key))
            log_ex_params["log" + key] = np.log(log_ex_params.pop(key))

        # Ensure true values are within confidence intervals
        synergy_assertions.assert_dict_values_in_intervals(log_ex_params, log_ci_95, tol=1e-5)

        # Ensure that less stringent CI is narrower
        # [=====95=====]  More confidence requires wider interval
        #     [===50==]   Less confidence but tighter interval
        confidence_intervals_50 = model.get_confidence_intervals(confidence_interval=50)
        synergy_assertions.assert_dict_interval_is_contained_in_other(confidence_intervals_50, confidence_intervals_95)


if __name__ == "__main__":
    unittest.main()

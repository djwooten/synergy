# TODO: Ensure proper behavior of bounds when fitting

import os
import sys
import unittest
from copy import deepcopy
from typing import Dict
from unittest import TestCase

import numpy as np

from synergy.combination import BRAID
from synergy.testing_utils import assertions as synergy_assertions
from synergy.testing_utils.test_data_loader import load_test_data
from synergy.utils import dose_utils

MAX_FLOAT = sys.float_info.max

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _get_E3(E0, E1, E2, beta):
    strongest_E = np.amin(np.asarray([E1, E2]), axis=0)
    return strongest_E - beta * (E0 - strongest_E)


class BRAIDTests(TestCase):
    """Tests for the 2D BRAID synergy model."""

    def test_is_specified_and_fit_at_creation(self):
        """Ensure expected behaviors of is_specified and is_fit when creating (but not fitting) models."""
        non_specified_model = BRAID()
        specified_model_kappa = BRAID(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, kappa=0, mode="kappa")
        specified_model_delta = BRAID(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, delta=1, mode="delta")
        specified_model_both = BRAID(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, kappa=0, delta=1, mode="both")
        nan_model = BRAID(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, kappa=np.nan)
        none_model = BRAID(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1)

        for model in [non_specified_model, nan_model, none_model]:
            self.assertFalse(model.is_specified)
            self.assertFalse(model.is_fit)

        for model in [specified_model_kappa, specified_model_delta, specified_model_both]:
            self.assertTrue(model.is_specified)
            self.assertFalse(model.is_fit)

    def test_no_scores_if_model_specified(self):
        """Ensure the model scores are not present if the model was not fit to data."""
        model = BRAID(E0=1, E1=0, E2=0, E3=0, h1=1, h2=2, C1=1, C2=1, kappa=0)
        self.assertFalse(hasattr(model, "aic"))
        self.assertFalse(hasattr(model, "bic"))
        self.assertFalse(hasattr(model, "r_squared"))
        self.assertFalse(hasattr(model, "sum_of_squares_residuals"))

    def test_asymptotic_limits(self):
        """Ensure model behaves well when d=0

        TODO: d -> inf, or d = C
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
        d1max = np.power(d1max, 0.6)
        d2max = np.power(d2max, 0.6)

        model = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=1)

        # Ensure E(0) == E0
        self.assertEqual(model.E(0, 0), E0, "E(0) should be E0")

        # Ensure E(inf, 0) == E1 and E(0, inf) == E2 and E(inf, inf) == E3
        # TODO: E(inf, inf) does not actually equal E3. What does it equal?
        # for E, d in zip([E1, E2, E3], [(d1max, 0), (0, d2max), (d1max, d2max)]):
        for E, d in zip([E1, E2], [(d1max, 0), (0, d2max)]):
            d1, d2 = d
            d1s = "0" if d1 == 0 else "inf"
            d2s = "0" if d2 == 0 else "inf"

            if np.abs(E) < 1e-6:  # For tiny values, compare E(inf) == Emax
                self.assertAlmostEqual(model.E(d1, d2), E, places=1, msg=f"E({d1s}, {d2s}) should be {E:0.3g}")
            else:  # For larger values, compare E(inf) / Emax ~= 1
                self.assertAlmostEqual(model.E(d1, d2) / E, 1.0, places=2, msg=f"E({d1s}, {d2s}) should be {E:0.3g}")

    def test_synergism(self):
        """Ensure synergistic parameters lead to stronger E"""
        E0, E1, E2, E3, h1, h2, C1, C2 = 1, 0.5, 0.3, 0.0, 1, 1, 1, 1
        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, n_points1=6, n_points2=6)

        for delta, kappa in [(1, 1), (2, 0), (2, 1)]:
            model = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=kappa, delta=delta, mode="both")
            E = model.E(d1, d2)
            reference = model.E_reference(d1, d2)
            self.assertTrue((E < reference).all())

    def test_antagonism(self):
        """Ensure antagonistic parameters lead to weaker E"""
        E0, E1, E2, E3, h1, h2, C1, C2 = 1, 0.5, 0.3, 0.0, 1, 1, 1, 1
        d1, d2 = dose_utils.make_dose_grid(C1 / 20, C1 * 20, C2 / 20, C2 * 20, n_points1=6, n_points2=6)
        for delta, kappa in [(1, -1), (0.5, 0), (0.5, -1)]:
            model = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=kappa, delta=delta, mode="both")
            E = model.E(d1, d2)
            reference = model.E_reference(d1, d2)
            self.assertTrue((E > reference).all())


class BRAIDFitTests(TestCase):
    """Tests requiring fitting the 2D BRAID synergy model."""

    EXPECTED_PARAMETERS: Dict[str, Dict[str, float]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.EXPECTED_PARAMETERS = {
            "synthetic_BRAID_reference_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 1,
                "kappa": 0,
            },
            "synthetic_BRAID_delta_synergy_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 2,
                "kappa": 0,
            },
            "synthetic_BRAID_delta_antagonism_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 0.5,
                "kappa": 0,
            },
            "synthetic_BRAID_kappa_synergy_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 1,
                "kappa": 1,
            },
            "synthetic_BRAID_kappa_antagonism_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 1,
                "kappa": -1,
            },
            "synthetic_BRAID_delta_kappa_synergy_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 2,
                "kappa": 1,
            },
            "synthetic_BRAID_delta_kappa_antagonism_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 0.5,
                "kappa": -1,
            },
            "synthetic_BRAID_asymmetric_1.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 0.5,
                "kappa": 1,
            },
            "synthetic_BRAID_asymmetric_2.csv": {
                "E0": 1,
                "E1": 0.5,
                "E2": 0.1,
                "E3": 0,
                "h1": 1,
                "h2": 1,
                "C1": 1,
                "C2": 1,
                "delta": 2,
                "kappa": -1,
            },
        }

    def test_BRAID_fit_no_bootstrap(self):
        """Ensure the model fits correctly."""
        for fname in [
            "synthetic_BRAID_reference_1.csv",
            "synthetic_BRAID_delta_synergy_1.csv",
            "synthetic_BRAID_delta_antagonism_1.csv",
            # "synthetic_BRAID_kappa_synergy_1.csv",
            "synthetic_BRAID_kappa_antagonism_1.csv",
            "synthetic_BRAID_delta_kappa_synergy_1.csv",
            # "synthetic_BRAID_delta_kappa_antagonism_1.csv",
        ]:
            np.random.seed(3402348)
            expected_parameters = deepcopy(self.EXPECTED_PARAMETERS[fname])
            mode = "both"
            if "delta" in fname and "kappa" not in fname:
                mode = "delta"
                expected_parameters.pop("kappa")
            elif "kappa" in fname and "delta" not in fname:
                mode = "kappa"
                expected_parameters.pop("delta")
            d1, d2, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
            model = BRAID(mode=mode)
            model.fit(d1, d2, E, use_jacobian=False)

            # Ensure the hill is fit
            self.assertTrue(model.is_specified)
            self.assertTrue(model.is_fit)
            self.assertTrue(model.is_converged)

            # Compare C, h, alpha, and gamma in log-scale
            observed_parameters = model.get_parameters()

            for key in ["delta", "h1", "h2", "C1", "C2"]:
                if key in expected_parameters and key in observed_parameters:
                    expected_parameters["log" + key] = np.log(expected_parameters.pop(key))
                    observed_parameters["log" + key] = np.log(observed_parameters.pop(key))

            # Ensure the parameters are approximately correct
            synergy_assertions.assert_dict_allclose(observed_parameters, expected_parameters, atol=0.6, err_msg=fname)

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

    def test_BRAID_fit_bootstrap(self):
        """Ensure confidence intervals work reasonably.

        Checks:
            1) model.bootstrap_parameters is not None
            2) True parameters are within 95% confidence interval (tol=1e-4)
            3) 50% confidence interval is contained entirely within 95% confidence interval
        """
        for fname in [
            "synthetic_BRAID_reference_1.csv",
            # "synthetic_BRAID_delta_synergy_1.csv",
            # "synthetic_BRAID_delta_antagonism_1.csv",
            # "synthetic_BRAID_kappa_synergy_1.csv",
            "synthetic_BRAID_kappa_antagonism_1.csv",
            "synthetic_BRAID_delta_kappa_synergy_1.csv",
            # "synthetic_BRAID_delta_kappa_antagonism_1.csv",
        ]:
            np.random.seed(8486435)
            expected_parameters = deepcopy(self.EXPECTED_PARAMETERS[fname])
            mode = "both"
            if "delta" in fname and "kappa" not in fname:
                mode = "delta"
                expected_parameters.pop("kappa")
            elif "kappa" in fname and "delta" not in fname:
                mode = "kappa"
                expected_parameters.pop("delta")
            d1, d2, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
            model = BRAID(mode=mode)
            model.fit(d1, d2, E, bootstrap_iterations=100, use_jacobian=False)

            # Ensure there were bootstrap iterations
            self.assertIsNotNone(model.bootstrap_parameters)

            confidence_intervals_95 = model.get_confidence_intervals()

            # Ensure true values are within confidence intervals
            synergy_assertions.assert_dict_values_in_intervals(
                expected_parameters, confidence_intervals_95, tol=0.1, err_msg=fname
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

# TODO: Ensure proper behavior of bounds when fitting

import os
import sys
import unittest
from copy import deepcopy
from unittest import TestCase

import hypothesis
import numpy as np
from hypothesis import given, seed
from hypothesis.strategies import floats, sampled_from

from synergy.single import Hill, Hill_2P, Hill_CI
from synergy.testing_utils.test_data_loader import load_test_data
from synergy.testing_utils import assertions as synergy_assertions


MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class HillTests(TestCase):
    """Tests for 1D Hill dose-response models."""

    MODEL: type[Hill]

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill

    def test_is_specified_and_fit_at_creation(self):
        """Ensure expected behaviors of is_specified and is_fit when creating (but not fitting) models."""
        non_specified_model = self.MODEL()

        if self.MODEL is Hill_CI:
            specified_model = self.MODEL(h=1, C=1)
            nan_model = self.MODEL(h=1, C=np.nan)
            none_model = self.MODEL(h=None, C=1)
        else:
            specified_model = self.MODEL(E0=1, Emax=0, h=1, C=1)
            nan_model = self.MODEL(E0=1, Emax=0, h=1, C=np.nan)
            none_model = self.MODEL(E0=1, Emax=0, h=None, C=1)

        for model in [non_specified_model, nan_model, none_model]:
            self.assertFalse(model.is_specified)
            self.assertFalse(model.is_fit)

        for model in [specified_model]:
            self.assertTrue(model.is_specified)
            self.assertFalse(model.is_fit)

    def test_no_scores_if_model_specified(self):
        """Ensure the model scores are not present if the model was not fit to data."""
        if self.MODEL is Hill_CI:
            model = self.MODEL(h=1, C=1)
        else:
            model = self.MODEL(E0=1, Emax=0, h=1, C=1)

        self.assertFalse(hasattr(model, "aic"))
        self.assertFalse(hasattr(model, "bic"))
        self.assertFalse(hasattr(model, "r_squared"))
        self.assertFalse(hasattr(model, "sum_of_squares_residuals"))

    @seed(0)
    @given(
        floats(  # E0
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10,
        ),
        floats(  # Emax
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10,
        ),
        # logh
        floats(allow_nan=False, allow_infinity=False, min_value=-2, max_value=2),
        # logC
        floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10),
    )
    def test_asymptotic_limits(self, E0: float, Emax: float, logh: float, logC: float):
        """Ensure model behaves well when d=0, d->inf, or d=C

        Constraints are placed on values to avoid:
            - Overflows for d**h + C**h
            - Overflows for E0 - Emax
            - Numerical problems (in the test) calculating differences for very large numbers
            - Numerical problems (in the test) calculating ratios of numbers near 0 (< 1e-6)
        """
        h = np.exp(logh)
        C = np.exp(logC)

        # This will get d**h + C**h as close to MAX_FLOAT as possible
        dmax = MAX_FLOAT ** (1 / max(h, 1.0)) - C**h
        dmax /= 1.1

        if self.MODEL is Hill_CI:
            model = self.MODEL(h=h, C=C)
            E0 = 1.0
            Emax = 0.0
        else:
            model = self.MODEL(E0=E0, Emax=Emax, h=h, C=C)
        d = np.asarray([0, C, dmax])
        E = model.E(d)

        # Ensure E(0) == E0
        self.assertEqual(E[0], E0, "E(0) should be E0")

        # Ensure E(inf) == Emax
        if np.abs(Emax) < 1e-6:  # For tiny values, compare E(inf) == Emax
            self.assertAlmostEqual(E[2], Emax, msg="E(inf) should be Emax")
        else:  # For larger values, compare E(inf) / Emax ~= 1
            self.assertAlmostEqual(E[2] / Emax, 1.0, places=3, msg="E(inf) should be Emax")

        # Ensure E(C) == (E0 + Emax) / 2
        if np.abs(E0 + Emax) < 1e-6:  # For tiny values, compare E(C) == (E0 + Emax) / 2.0
            self.assertAlmostEqual(E[1], (E0 + Emax) / 2.0, msg="E(C) should be haflway between E0 and Emax")
        else:  # For larger values, compare the ratio ~= 1
            self.assertAlmostEqual(
                (E0 + Emax) / (2.0 * E[1]), 1.0, places=3, msg="E(C) should be haflway between E0 and Emax"
            )

    @seed(1)
    @given(
        floats(  # E0
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10,
        ),
        floats(  # Emax
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10,
        ),
        # logh
        floats(allow_nan=False, allow_infinity=False, min_value=-1, max_value=1),
        # logC
        floats(allow_nan=False, allow_infinity=False, min_value=-7, max_value=7),
    )
    def test_inverse(self, E0: float, Emax: float, logh: float, logC: float):
        """Ensure E_inv() successfully inverts"""
        # Skip tests when E0 ~= Emax, because it is impossible to invert a flat line
        if np.abs(E0 - Emax) < 1e-6:
            return

        h = np.exp(logh)
        C = np.exp(logC)
        if self.MODEL is Hill_CI:
            model = self.MODEL(h=h, C=C)
            E0 = 1.0
            Emax = 0.0
        else:
            model = self.MODEL(E0=E0, Emax=Emax, h=h, C=C)
        d = np.logspace(logC - 1, logC + 1)
        E = model.E(d)
        E_inv = model.E_inv(E)
        np.testing.assert_allclose(E_inv, d, rtol=0.01, err_msg="E_inv(E(d)) should equal d")


class HillFitTests(TestCase):
    """Tests requiring fitting 1D Hill dose-response models."""

    MODEL: type[Hill]
    INIT_KWARGS: dict
    EXPECTED_PARAMETERS: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill
        cls.INIT_KWARGS = {}
        cls.EXPECTED_PARAMETERS = {"synthetic_hill_1.csv": {"E0": 1.0, "Emax": 0.0, "h": 1.0, "C": 1.0}}

    @hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_hill_fit_no_bootstrap(self, fname):
        """Ensure the model fits correctly."""
        expected_parameters = self.EXPECTED_PARAMETERS[fname]

        d, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
        model = self.MODEL(**self.INIT_KWARGS)
        model.fit(d, E)

        # Ensure the hill is fit
        self.assertTrue(model.is_specified)
        self.assertTrue(model.is_fit)
        self.assertTrue(model.is_converged)

        # Ensure the parameters are approximately correct
        print(model.get_parameters())
        print(expected_parameters)
        synergy_assertions.assert_dict_allclose(model.get_parameters(), expected_parameters, atol=0.2)

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

    @hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_hill_fit_bootstrap(self, fname):
        """Ensure confidence intervals work reasonably."""
        expected_parameters = self.EXPECTED_PARAMETERS[fname]

        d, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
        model = self.MODEL(**self.INIT_KWARGS)
        model.fit(d, E, bootstrap_iterations=100)

        # Ensure there were bootstrap iterations
        self.assertIsNotNone(model.bootstrap_parameters)

        confidence_intervals_95 = model.get_confidence_intervals()

        # Ensure true values are within confidence intervals
        synergy_assertions.assert_dict_values_in_intervals(expected_parameters, confidence_intervals_95)

        # Ensure that less stringent CI is narrower
        # [=====95=====]  More confidence requires wider interval
        #     [===50==]   Less confidence but tighter interval
        confidence_intervals_50 = model.get_confidence_intervals(confidence_interval=50)
        synergy_assertions.assert_dict_interval_is_contained_in_other(confidence_intervals_50, confidence_intervals_95)

    @hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_dose_scale(self, fname):
        """Ensure confidence intervals work reasonably."""
        d, E = load_test_data(os.path.join(TEST_DATA_DIR, fname))
        scale = 1e9
        d *= scale
        expected = deepcopy(self.EXPECTED_PARAMETERS[fname])
        expected["C"] *= scale
        expected_C = expected["C"]

        model = self.MODEL(C_bounds=(expected_C / 10.0, expected_C * 10.0), **self.INIT_KWARGS)

        # Ensure the parameters are approximately correct
        model.fit(d, E)
        observed = model.get_parameters()

        # Do C differently from everything else, since it is on a different scale
        synergy_assertions.assert_dict_allclose(
            {k: observed[k] for k in observed.keys() if k != "C"},
            {k: expected[k] for k in expected.keys() if k != "C"},
            atol=0.2,
        )
        np.testing.assert_allclose(observed["C"], expected["C"], atol=0.2 * scale)


class Hill2PTests(HillTests):
    """Tests for 1D Hill_2P dose-response models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill_2P


class Hill2PFitTests(HillFitTests):
    """-"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill_2P
        cls.INIT_KWARGS = {"E0": 1.0, "Emax": 0.0}
        cls.EXPECTED_PARAMETERS = {"synthetic_hill_1.csv": {"h": 1.0, "C": 1.0}}


class HillCITests(HillTests):
    """Tests for the combination index model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill_CI


class HillCIFitTests(HillFitTests):
    """-"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill_CI
        cls.INIT_KWARGS = {}
        cls.EXPECTED_PARAMETERS = {"synthetic_hill_1.csv": {"h": 1.0, "C": 1.0}}

    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_hill_fit_bootstrap(self, fname):
        """TODO: Bootstrap resampling is not yet implemented for Hill_CI"""
        pass


if __name__ == "__main__":
    unittest.main()

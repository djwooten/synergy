import os
import sys
import unittest
from unittest import TestCase

import hypothesis
import numpy as np
from hypothesis import given, seed
from hypothesis.strategies import floats, sampled_from

from synergy.single.hill import Hill, Hill_2P, Hill_CI
from synergy.testing_utils.synthetic_data_loader import load_synthetic_data


MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class HillTests(TestCase):
    """Tests for 1D Hill dose-response models."""

    MODEL: type[Hill]

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill

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
        cls.EXPECTED_PARAMETERS = {"synthetic_hill_1.csv": [1.0, 0.0, 1.0, 1.0]}

    @hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_hill_fit_no_bootstrap(self, fname):
        """Ensure the model fits correctly."""
        expected_parameters = self.EXPECTED_PARAMETERS[fname]

        d, E = load_synthetic_data(os.path.join(TEST_DATA_DIR, fname))
        hill = self.MODEL(**self.INIT_KWARGS)
        hill.fit(d, E)

        # Ensure the hill is fit
        self.assertTrue(hill.is_specified)

        # Ensure the parameters are approximately correct
        observed = np.asarray(hill.get_parameters())
        expected = np.asarray(expected_parameters)
        np.testing.assert_allclose(observed, expected, atol=0.2)

        # Ensure the scores were calculated
        self.assertIsNotNone(hill.aic)
        self.assertIsNotNone(hill.bic)
        self.assertGreaterEqual(hill.r_squared, 0, msg="r_squared should be between 0 and 1")
        self.assertLessEqual(hill.r_squared, 1, msg="r_squared should be less than 1")
        self.assertGreaterEqual(hill.sum_of_squares_residuals, 0, msg="sum_of_squares_residuals should be >= 0")

        # Ensure there were no bootstrap iterations
        self.assertIsNone(hill.bootstrap_parameters)
        with self.assertRaises(ValueError):
            _ = hill.get_confidence_intervals()

    @hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.differing_executors])
    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_hill_fit_bootstrap(self, fname):
        """Ensure confidence intervals work reasonably."""
        expected_parameters = self.EXPECTED_PARAMETERS[fname]

        d, E = load_synthetic_data(os.path.join(TEST_DATA_DIR, fname))
        hill = self.MODEL(**self.INIT_KWARGS)
        hill.fit(d, E, bootstrap_iterations=100)

        # Ensure there were bootstrap iterations
        self.assertIsNotNone(hill.bootstrap_parameters)

        # Ensure true values are within confidence intervals
        confidence_intervals_95 = hill.get_confidence_intervals()
        for interval, true_val in zip(confidence_intervals_95, expected_parameters):
            self.assertTrue(interval[0] <= true_val <= interval[1])

        # Ensure that less stringent CI is narrower
        # [=====95=====]  More confidence requires wider interval
        #     [===50==]   Less confidence but tighter interval
        confidence_intervals_50 = hill.get_confidence_intervals(confidence_interval=50)
        for interval_95, interval_50 in zip(confidence_intervals_95, confidence_intervals_50):
            self.assertLessEqual(interval_95[0], interval_50[0])
            self.assertGreaterEqual(interval_95[1], interval_50[1])


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
        cls.EXPECTED_PARAMETERS = {"synthetic_hill_1.csv": [1.0, 1.0]}


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
        cls.EXPECTED_PARAMETERS = {"synthetic_hill_1.csv": [1.0, 1.0]}

    @given(sampled_from(["synthetic_hill_1.csv"]))
    def test_hill_fit_bootstrap(self, fname):
        """TODO: Bootstrap resampling is not yet implemented for Hill_CI"""
        pass


if __name__ == "__main__":
    unittest.main()

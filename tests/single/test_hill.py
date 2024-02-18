import sys
import unittest
from unittest import TestCase

from hypothesis import given, seed
from hypothesis.strategies import floats
import numpy as np

from synergy.single.parametric_base import ParameterizedModel1D
from synergy.single.hill import Hill, Hill_2P, Hill_CI


MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min


class BaseHillTest(TestCase):
    """Base class defining basic values used by all tests"""

    dlog = np.logspace(-3, 3, 12)
    dlin = np.linspace(0, 3, 12)


class HillTests(BaseHillTest):
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

        hill = self.MODEL(E0=E0, Emax=Emax, h=h, C=C)
        d = np.asarray([0, C, dmax])
        E = hill.E(d)

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
        """Ensure inverse successfully inverts"""
        # Skip tests when E0 ~= Emax, because it is impossible to invert a flat line
        if np.abs(E0 - Emax) < 1e-6:
            return

        h = np.exp(logh)
        C = np.exp(logC)
        hill = self.MODEL(E0=E0, Emax=Emax, h=h, C=C)
        d = np.logspace(logC - 1, logC + 1)
        E = hill.E(d)
        E_inv = hill.E_inv(E)
        np.testing.assert_allclose(E_inv, d, rtol=0.01, err_msg="E_inv(E(d)) should equal d")


class HillFitTests(BaseHillTest):
    """Tests requiring fitting 1D Hill dose-response models."""


class Hill1PTests(HillTests):
    """Tests for 1D Hill_2P dose-response models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = Hill_2P


if __name__ == "__main__":
    unittest.main()

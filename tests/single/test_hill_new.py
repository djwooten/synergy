# TODO: Ensure proper behavior of bounds when fitting

import math
import os
import sys

import numpy as np
from hypothesis import given, seed
from hypothesis.strategies import floats

from synergy.single.hill import Hill, Hill_CI


MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestHill:
    """Tests for 1D Hill dose-response models."""

    MODEL: type[Hill] = Hill

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
            assert not model.is_specified
            assert not model.is_fit

        for model in [specified_model]:
            assert model.is_specified
            assert not model.is_fit

    def test_no_scores_if_model_specified(self):
        """Ensure the model scores are not present if the model was not fit to data."""
        if self.MODEL is Hill_CI:
            model = self.MODEL(h=1, C=1)
        else:
            model = self.MODEL(E0=1, Emax=0, h=1, C=1)

        assert not hasattr(model, "aic")
        assert not hasattr(model, "bic")
        assert not hasattr(model, "r_squared")
        assert not hasattr(model, "sum_of_squares_residuals")

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
        assert E[0] == E0

        # Ensure E(inf) == Emax
        if np.abs(Emax) < 1e-6:  # For tiny values, compare E(inf) == Emax
            assert math.isclose(E[2], Emax, abs_tol=1e-6), "E(inf) should be Emax"
        else:  # For larger values, compare E(inf) / Emax ~= 1
            assert math.isclose(E[2] / Emax, 1.0), "E(inf) should be Emax"

        # Ensure E(C) == (E0 + Emax) / 2
        if np.abs(E0 + Emax) < 1e-6:  # For tiny values, compare E(C) == (E0 + Emax) / 2.0
            assert math.isclose(E[1], (E0 + Emax) / 2.0), "E(C) should be haflway between E0 and Emax"
        else:  # For larger values, compare the ratio ~= 1
            assert math.isclose((E0 + Emax) / (2.0 * E[1]), 1.0), "E(C) should be haflway between E0 and Emax"

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

import unittest
from unittest import TestCase

import numpy as np

from synergy.utils.model_mixins import ParametricModelMixins


class MagicMock:
    """Class to serve as a mock model"""


class TestParametricModelMixins(TestCase):
    """Test for model mixins"""

    def test_set_init_parameters(self):
        """Ensure parameters are set"""
        mock = MagicMock()
        ParametricModelMixins.set_init_parameters(mock, ["a", "b"], a=1, b=2)
        self.assertEqual(mock.a, 1)
        self.assertEqual(mock.b, 2)

    def test_set_parameters(self):
        """Ensure parameters are set"""
        mock = MagicMock()
        ParametricModelMixins.set_parameters(mock, ["a", "b"], 1, 2)
        self.assertEqual(mock.a, 1)
        self.assertEqual(mock.b, 2)

    def test_make_summary_row_with_ci(self):
        """Test make_summary_row giving it a confidence interval"""
        param = "parameter"
        ci = {"parameter": (0.5, 2)}
        val = 1.0
        row = ParametricModelMixins.make_summary_row(param, 0, val, ci, 0, False, "synergistic", "antagonistic")
        rowstr = " ".join(row)
        self.assertEqual(rowstr, "parameter 1 (0.5, 2) > 0 synergistic")

    def test_make_summary_row_no_ci(self):
        """Test make_summary_row without a confidence interval"""
        param = "parameter"
        val = 1.0
        row = ParametricModelMixins.make_summary_row(param, 0, val, None, 0, False, "synergistic", "antagonistic")
        rowstr = " ".join(row)
        self.assertEqual(rowstr, "parameter 1 > 0 synergistic")

    def test_find_matching_parameter(self):
        """Ensure matching parameter is found correctly"""
        self.assertEqual(
            ParametricModelMixins._find_matching_parameter(["ele", "elephant", "gooses", "gopher"], "ele"),
            "ele",
        )

        self.assertEqual(
            ParametricModelMixins._find_matching_parameter(["ele", "elephant", "gooses", "gopher"], "elephant"),
            "elephant",
        )

        self.assertEqual(
            ParametricModelMixins._find_matching_parameter(["ele", "elephant", "gooses", "gopher"], "special"),
            "",
        )

    def test_set_specific_bounds(self):
        """Ensure bounds are set when explicitly passed"""
        mock = MagicMock()
        parameters = ["not_specified", "default_specified", "explicit_specified"]
        transform = lambda x: x  # no log-scaling for this test  # noqa E731
        default_bounds = {"default_specified": (1, 2)}
        ParametricModelMixins.set_bounds(mock, transform, default_bounds, parameters, explicit_specified_bounds=(3, 4))
        self.assertEqual(mock._bounds, ([-np.inf, 1, 3], [np.inf, 2, 4]))

    def test_transform_bounds(self):
        """Ensure bounds are transformed correctly"""
        mock = MagicMock()
        parameters = ["log"]
        transform = lambda x: np.log10(x)  # noqa E731
        ParametricModelMixins.set_bounds(mock, transform, {}, parameters, log_bounds=(1e-1, 10))
        self.assertEqual(mock._bounds, ([-1.0], [1.0]))

    def test_set_generic_bounds(self):
        """Ensure setting generic bounds (such as E_bounds) works"""
        mock = MagicMock()
        parameters = ["E0", "E1", "E2", "E3", "alpha1", "alpha2"]
        transform = lambda x: x  # noqa E731
        ParametricModelMixins.set_bounds(
            mock, transform, {}, parameters, E_bounds=(0, 1), E0_bounds=(0.9, 1.1), alpha_bounds=(2, 3)
        )
        self.assertEqual(mock._bounds, ([0.9, 0, 0, 0, 2, 2], [1.1, 1, 1, 1, 3, 3]))

    def test_get_generic_parameter(self):
        """Ensure generic parameter is found correctly"""
        generic_bounds = {"E": (1, 2), "alpha": (3, 4), "Extra": (5, 6)}
        self.assertEqual(ParametricModelMixins._get_generic_parameter(generic_bounds, "E_0"), "E")
        self.assertEqual(ParametricModelMixins._get_generic_parameter(generic_bounds, "E_1"), "E")
        self.assertEqual(ParametricModelMixins._get_generic_parameter(generic_bounds, "E_1,2,3"), "E")
        self.assertEqual(ParametricModelMixins._get_generic_parameter(generic_bounds, "Extra_0"), "Extra")
        self.assertEqual(ParametricModelMixins._get_generic_parameter(generic_bounds, "alpha_1,2_3"), "alpha")
        self.assertEqual(ParametricModelMixins._get_generic_parameter(generic_bounds, "gamma_0"), "")

    def test_get_bound(self):
        """Ensure the correct bound is returned for a parameter"""
        generic_bounds = {"E": (1, 2), "alpha": (3, 4)}
        default_bounds = {"alpha_1": (3.5, 3.6), "x_1": (7, 8)}
        kwargs = {"E0_bounds": (0.9, 1.1)}
        # Get specific param (E0) passed in kwargs - specific takes precedence over generic
        self.assertEqual(ParametricModelMixins._get_bound("E0", generic_bounds, default_bounds, **kwargs), (0.9, 1.1))
        # Get generic param (E) for E1
        self.assertEqual(ParametricModelMixins._get_bound("E1", generic_bounds, default_bounds, **kwargs), (1, 2))
        # Get generic param (alpha) for alpha_1 - generic takes precedence over default
        self.assertEqual(ParametricModelMixins._get_bound("alpha_1", generic_bounds, default_bounds, **kwargs), (3, 4))
        # Get default param for x_1
        self.assertEqual(ParametricModelMixins._get_bound("x_1", generic_bounds, default_bounds, **kwargs), (7, 8))
        # Nothing is set for x, so it should return infinite bounds
        self.assertEqual(
            ParametricModelMixins._get_bound("x", generic_bounds, default_bounds, **kwargs), (-np.inf, np.inf)
        )


if __name__ == "__main__":
    unittest.main()

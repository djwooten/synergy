import numpy as np
import pytest

from synergy import single, utils
from synergy.exceptions import InvalidDrugModelError


class TestSanitizeInitialGuess:
    """Tests for the sanitize_initial_guess function."""

    def test_p0_none(self):
        """Ensure if p0 is None, it is replaced with the mean of the bounds"""
        p0 = [None] * 6
        bounds = [
            (-1, 1),  # 0
            (-np.inf, -10),  # -100
            (-np.inf, 10),  # 0
            (-10, np.inf),  # 0
            (10, np.inf),  # 100
            (-np.inf, np.inf),  # 0
        ]
        bounds = tuple(zip(*bounds))  # convert to the expected format of (lower, upper)
        sanitized = utils.sanitize_initial_guess(p0, bounds)
        assert sanitized == [0, -100, 0, 0, 100, 0]

    def test_p0_in_bounds(self):
        """Ensure if p0 is within the bounds, it is not changed"""
        p0 = [0, -100, 0, 0, 100, 0]
        bounds = [
            (-1, 1),  # 0
            (-np.inf, -10),  # -100
            (-np.inf, 10),  # 0
            (-10, np.inf),  # 0
            (10, np.inf),  # 100
            (-np.inf, np.inf),  # 0
        ]
        bounds = tuple(zip(*bounds))
        sanitized = utils.sanitize_initial_guess(p0, bounds)
        assert sanitized == p0

    def test_p0_out_of_bounds(self):
        """Ensure if p0 is outside the bounds, it is replaced with the bounds"""
        p0 = [2, -200, 20, 20, -20, 20]
        bounds = [
            (-1, 1),  # above
            (-np.inf, -10),
            (-np.inf, 10),  # above
            (-10, np.inf),
            (10, np.inf),  # below
            (-np.inf, np.inf),
        ]
        bounds = tuple(zip(*bounds))
        sanitized = utils.sanitize_initial_guess(p0, bounds)
        assert sanitized == [1, -200, 10, 20, 10, 20]


class TestSanitizeSingleDrugModels:
    """Tests for the sanitize_single_drug_model function."""

    def test_sanitize_model_with_none(self):
        """Ensure if model is None, an instance of the default type is returned"""
        model = None
        sanitized = utils.sanitize_single_drug_model(model, single.Hill, single.DoseResponseModel1D)
        assert isinstance(sanitized, single.Hill)

        sanitized = utils.sanitize_single_drug_model(model, single.LogLinear, single.DoseResponseModel1D)
        assert isinstance(sanitized, single.LogLinear)

    def test_sanitize_model_with_object(self):
        """Ensure if model is a valid object, it is returned as is"""
        model = single.Hill()
        sanitized = utils.sanitize_single_drug_model(model, single.Hill, single.DoseResponseModel1D)
        assert sanitized == model

    def test_sanitize_model_with_class(self):
        """Ensure if model is a valid class, an instance of that class is returned"""
        model = single.Hill_2P
        sanitized = utils.sanitize_single_drug_model(model, single.Hill, single.DoseResponseModel1D)
        assert isinstance(sanitized, single.Hill_2P)

    def test_sanitize_model_with_invalid_class(self):
        """Ensure if model is an invalid class, an exception is raised"""
        model = single.LogLinear
        with pytest.raises(InvalidDrugModelError):
            utils.sanitize_single_drug_model(model, single.Hill, single.Hill)

    def test_sanitize_model_with_invalid_object(self):
        """Ensure if model is an invalid object, an exception is raised"""
        model = single.LogLinear()
        with pytest.raises(InvalidDrugModelError):
            utils.sanitize_single_drug_model(model, single.Hill, single.Hill)


def test_format_table():
    """Ensure the format_table function returns a properly formatted table"""
    rows = [
        ["a", "boo", "c", "d"],
        ["1", "2", "300", "4"],
        ["4", "5", "6", "7"],
    ]
    expected = """
a  |  boo  |  c    |  d
=======================
1  |  2    |  300  |  4
4  |  5    |  6    |  7
"""
    table = utils.format_table(rows)
    assert table == expected[1:-1]

    expected = """
a  |  boo  |  c    |  d
1  |  2    |  300  |  4
4  |  5    |  6    |  7
"""
    table = utils.format_table(rows, first_row_is_header=False)
    assert table == expected[1:-1]

    expected = """
a;boo;c  ;d
===========
1;2  ;300;4
4;5  ;6  ;7
"""
    table = utils.format_table(rows, col_sep=";")
    assert table == expected[1:-1]

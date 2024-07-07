#    Copyright (C) 2020 David J. Wooten
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import inspect
from typing import Callable, Sequence, Tuple

import numpy as np

from synergy.exceptions import InvalidDrugModelError


def residual_ss(d1, d2, E, function: Callable):
    """Calculate the sum of squares of the residuals for a 2D dose response model.

    :param ArrayLike d1: The doses of drug 1
    :param ArrayLike d2: The doses of drug 2
    :param ArrayLike E: The observed values
    :param Callable function: The model to use
    """
    E_model = function(d1, d2)
    return np.sum((E - E_model) ** 2)


def residual_ss_1d(d, E, function: Callable):
    """Calculate the sum of squares of the residuals for a 1D dose response model.

    :param ArrayLike d: The doses
    :param ArrayLike E: The observed values
    :param Callable function: The model to use
    """
    E_model = function(d)
    return np.sum((E - E_model) ** 2)


def AIC(sum_of_squares_residuals: float, n_parameters: int, n_samples: int) -> float:
    """Calculate the Akaike Information Criterion.

    SOURCE: AIC under the Framework of Least Squares Estimation, HT Banks, Michele L Joyner, 2017
    Equations (6) and (16)
    https://projects.ncsu.edu/crsc/reports/ftp/pdf/crsc-tr17-09.pdf

    :param float sum_of_squares_residuals: The sum of squares of the residuals
    :param int n_parameters: The number of parameters in the model
    :param int n_samples: The number of samples
    :return float: The AIC value
    """
    aic = n_samples * np.log(sum_of_squares_residuals / n_samples) + 2 * (n_parameters + 1)
    if n_samples / n_parameters > 40:
        return aic
    else:
        return aic + 2 * n_parameters * (n_parameters + 1) / (n_samples - n_parameters - 1)


def BIC(sum_of_squares_residuals: float, n_parameters: int, n_samples: int) -> float:
    """Calculate the Bayesian Information Criterion

    :param float sum_of_squares_residuals: The sum of squares of the residuals
    :param int n_parameters: The number of parameters in the model
    :param int n_samples: The number of samples
    :return float: The BIC value
    """
    return n_samples * np.log(sum_of_squares_residuals / n_samples) + (n_parameters + 1) * np.log(n_samples)


def r_squared(E, sum_of_squares_residuals: float) -> float:
    """Calculate the R^2 value.

    :param ArrayLike E: The observed values
    :param float sum_of_squares_residuals: The sum of squares of the residuals
    :return float: The R^2 value
    """
    ss_tot = np.sum((E - np.mean(E)) ** 2)
    return 1 - sum_of_squares_residuals / ss_tot


def sanitize_initial_guess(p0, bounds: Tuple[Sequence[float], Sequence[float]]):
    """Ensure sure p0 is within the bounds.

    :param p0: Initial guess for the parameters
    :param bounds: Lower and upper bounds for the parameters
    :return: The sanitized initial guess
    """
    index = 0
    for x, lower, upper in zip(p0, *bounds):
        if x is None:
            # (-inf, inf): use 0
            if np.isinf(lower) and np.isinf(upper):
                p0[index] = 0

            # (-inf, u): use 0 or 10 * u
            elif np.isinf(lower):
                if upper > 0:  # (-inf, +u): use 0 because it is within the bounds
                    p0[index] = 0
                else:  # (-inf, -u): use -u * 10 so that we are not right at the boundary (e.g., (-inf, -1) -> -10)
                    p0[index] = upper * 10

            # (l, +inf): use 0 or 10 * x
            elif np.isinf(upper):
                if lower > 0:  # (+l, +inf): use l * 10 so we are not right at the boundary (e.g., (1, +inf) -> 10)
                    p0[index] = lower * 10
                else:  # (-l, +inf): use 0 because it is within the bounds
                    p0[index] = 0

            # (x, y): use the midpoint
            else:
                p0[index] = np.mean((lower, upper))

        elif x < lower:
            p0[index] = lower
        elif x > upper:
            p0[index] = upper
        index += 1
    return p0


def sanitize_single_drug_model(model, default_type: type, required_type: type, **kwargs):
    """Ensure the given single drug model is a class or object of a class that is permitted for the given synergy model.

    :param DoseResponseModel1D model: The single drug model
    :param type default_type: The type of model to return if the given model is None
    :param type required_type: The class the model is expected to be an instance of
    :param kwargs: Additional arguments to pass to the model constructor
    :return DoseResponseModel1D: An object that is an instance of required_type
    """
    if model is None:
        model = default_type(**kwargs)

    if inspect.isclass(model):
        if required_type and not issubclass(model, required_type):
            raise InvalidDrugModelError(f"Expected a single drug model inheriting type {required_type.__name__}")
        model = model(**kwargs)

    elif required_type and not isinstance(model, required_type):
        raise InvalidDrugModelError(f"Expected a single drug model inheriting type {required_type.__name__}")

    return model


def format_table(rows: Sequence[Sequence[str]], first_row_is_header: bool = True, col_sep: str = "  |  ") -> str:
    """Format a list of rows into a human readable table.

    :param Sequence[Sequence[str]] rows: A list of rows, where each row is a list of strings
    :param bool first_row_is_header: Whether the first row should be formatted as a header
    :param str col_sep: The separator between columns
    :return str: A string representation of the table
    """
    if not rows:
        return ""

    num_columns = len(rows[0])
    max_column_widths = [max(*[len(row[col]) for row in rows]) for col in range(num_columns)]
    row_format = col_sep.join(["{:<%d}" % width for width in max_column_widths])

    row_strings = [row_format.format(*row) for row in rows]

    if first_row_is_header:
        rowsep = "=" * (sum(max_column_widths) + len(col_sep) * (num_columns - 1))
        row_strings = [row_strings[0]] + [rowsep] + row_strings[1:]

    return "\n".join(row_strings)

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
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from synergy.exceptions import InvalidDrugModelError


def remove_zeros(d, min_buffer=0.2, num_dilutions: int = 1):
    """Replace zeros with some semi-intelligently chosen small value

    When plotting on a log scale, 0 doses can cause problems. This replaces all 0's using the dilution factor between
    the smallest non-zero, and second-smallest non-zero doses. If that dilution factor is too close to 1, it will
    replace 0's doses with a dose that is min_buffer*(max(d)-min(d[d>0])) less than min(d[d>0]) on a log scale.

    Parameters
    ----------
    d : array_like
        Doses to remove zeros from. Original array will not be changed.

    min_buffer : float , default=0.2
        For very large dose arrays with very small step sizes (useful for getting smooth plots), replacing 0's may lead
        to a value too close to the smallest non-zero dose. min_buffer is the minimum buffer (in log scale, relative to
        the full dose range) that 0's will be replaced with.
    """

    d = np.array(d, copy=True)
    dmin = np.min(d[d > 0])  # smallest nonzero dose
    dmin2 = np.min(d[d > dmin])
    dilution = dmin / dmin2

    dmax = np.max(d)
    logdmin = np.log(dmin)
    logdmin2 = np.log(dmin2)
    logdmax = np.log(dmax)

    if (logdmin2 - logdmin) / (logdmax - logdmin) < min_buffer:
        logdmin2_effective = logdmin + min_buffer * (logdmax - logdmin)
        dilution = dmin / np.exp(logdmin2_effective)

    d[d == 0] = dmin * np.float_power(dilution, num_dilutions)
    return d


def residual_ss(d1, d2, E, model):
    E_model = model(d1, d2)
    return np.sum((E - E_model) ** 2)


def residual_ss_1d(d, E, model):
    E_model = model(d)
    return np.sum((E - E_model) ** 2)


def AIC(sum_of_squares_residuals, n_parameters, n_samples):
    """
    SOURCE: AIC under the Framework of Least Squares Estimation, HT Banks, Michele L Joyner, 2017
    Equations (6) and (16)
    https://projects.ncsu.edu/crsc/reports/ftp/pdf/crsc-tr17-09.pdf
    """
    aic = n_samples * np.log(sum_of_squares_residuals / n_samples) + 2 * (n_parameters + 1)
    if n_samples / n_parameters > 40:
        return aic
    else:
        return aic + 2 * n_parameters * (n_parameters + 1) / (n_samples - n_parameters - 1)


def BIC(sum_of_squares_residuals, n_parameters, n_samples):
    return n_samples * np.log(sum_of_squares_residuals / n_samples) + (n_parameters + 1) * np.log(n_samples)


def r_squared(E, sum_of_squares_residuals):
    ss_tot = np.sum((E - np.mean(E)) ** 2)
    return 1 - sum_of_squares_residuals / ss_tot


def sanitize_initial_guess(p0: ArrayLike, bounds: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
    """Ensure sure p0 is within the bounds.

    :param p0: Initial guess for the parameters
    :param bounds: Lower and upper bounds for the parameters
    :return: The sanitized initial guess
    """
    index = 0
    for x, lower, upper in zip(p0, *bounds):
        if x is None:
            if True in np.isinf((lower, upper)):
                p0[index] = np.min((np.max((0, lower)), upper))
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

    :param DRModel1D model: The single drug model
    :param type default_type: The type of model to return if the given model is None
    :param type required_type: The class the model is expected to be an instance of
    :param kwargs: Additional arguments to pass to the model constructor
    :return DRModel1D: An object that is an instance of required_type
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


def format_table(rows, first_row_is_header: bool = True, col_sep: str = "  |  ") -> str:
    """Format a list of rows into a human readable table.

    :param list[list[str]] rows: A list of rows, where each row is a list of strings
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

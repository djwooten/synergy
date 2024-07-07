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

import logging
from itertools import product
from typing import Sequence, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)


def remove_zeros(d, min_buffer: float = 0.2, num_dilutions: int = 1):
    """Replace zeros with a small value based on the dilution factor between the smallest non-zero doses.

    When plotting on a log scale, 0 doses can cause problems. This replaces all 0's using the dilution factor between
    the smallest non-zero, and second-smallest non-zero doses. If that dilution factor is too close to 1, it will
    replace 0's doses with a dose that is min_buffer*(max(d)-min(d[d>0])) less than min(d[d>0]) on a log scale.

    Parameters
    ----------
    d
        Doses to remove zeros from. Original array will not be changed.

    min_buffer : float , default=0.2
        For high resolution dose arrays with very small step sizes (useful for getting smooth plots), replacing 0's
        based on the dilution factor may lead to a value too close to the smallest non-zero dose. min_buffer is the
        minimum buffer (in log scale, relative to the full dose range) that 0's will be replaced with.
    """
    d = np.array(d, copy=True, dtype=np.float64)
    dmin = np.min(d[d > 0])  # smallest nonzero dose
    dmin2 = np.min(d[d > dmin])  # second smallest nonzero dose
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


def make_dose_grid(
    d1min: float,
    d1max: float,
    d2min: float,
    d2max: float,
    n_points1: int,
    n_points2: int,
    replicates: int = 1,
    logscale: bool = True,
    include_zero: bool = False,
):
    """Create a grid of doses.

    If ```logscale``` is `True`, use ```include_zero=True``` instead of setting the min dose to `0`.

    :param float d1min: Minimum dose for drug 1
    :param float d1max: Maximum dose for drug 1
    :param float d2min: Minimum dose for drug 2
    :param float d2max: Maximum dose for drug 2
    :param int n_points1: Number of distinct doses to include for drug 1
    :param int n_points2: Number of distinct doses to include for drug 2
    :param int replicates: The number of replicates to include for each dose combination
    :param bool logscale: If True, doses will be uniform in log space. If False, doses will be uniform in linear space.
    :param bool include_zero: If True, will include a dose of 0. (Only used if ```logscale``` is `True`)
    :return: (d1, d2)
    :rtype: tuple
    """
    if d1max <= d1min or d2max <= d2min:
        raise ValueError("The maximum dose must be higher than the minimum dose")
    if d1min < 0 or d2min < 0:
        raise ValueError("Doses must be non-negative")

    if logscale:
        if d1min == 0 or d2min == 0:
            raise ValueError(
                "Cannot generate doses on logscale with minimum dose of 0. Use `include_zero=True` instead."
            )
        if include_zero:  # 0 is handled separately for logscale dose grids
            n_points1 -= 1
            n_points2 -= 1
        d1 = np.logspace(np.log10(d1min), np.log10(d1max), num=n_points1)
        d2 = np.logspace(np.log10(d2min), np.log10(d2max), num=n_points2)
    else:
        d1 = np.linspace(d1min, d1max, num=n_points1)
        d2 = np.linspace(d2min, d2max, num=n_points2)

    if include_zero and logscale:
        d1 = np.append(0, d1)
        d2 = np.append(0, d2)

    D1, D2 = np.meshgrid(d1, d2)
    D1 = [D1.flatten()]
    D2 = [D2.flatten()]

    D1 = np.hstack(D1 * replicates)
    D2 = np.hstack(D2 * replicates)

    return D1, D2


def make_dose_grid_multi(
    dmin: Sequence[float],
    dmax: Sequence[float],
    npoints: Sequence[int],
    logscale: bool = True,
    include_zero: bool = False,
    replicates: int = 1,
) -> np.ndarray:
    """Create a grid of doses for N drugs.

    :param Sequence[float] dmin: Sequence of minimum doses for each drug
    :param Sequence[float] dmax: Sequence of maximum doses for each drug
    :param Sequence[int] npoints: Sequence of number of distinct doses to include for each drug
    :param bool logscale: If True, doses will be uniform in log space. If False, doses will be uniform in linear space.
    :param bool include_zero: If True, will include a dose of 0
    :param int replicates: The number of replicates to include for each dose combination
    :return np.ndarray: Dose grid
    """
    if not (len(dmin) == len(dmax) and len(dmin) == len(npoints)):
        raise ValueError("Cannot generate Nd drug grid with unequal N.")
    doses = []
    for Dmin, Dmax, n in zip(dmin, dmax, npoints):
        if include_zero and Dmin > 0:
            n -= 1
        if Dmax <= Dmin:
            raise ValueError(f"The maximum dose {Dmax} must be higher than the minimum dose {Dmin}")
        if logscale:
            if Dmin == 0:
                raise ValueError(
                    "Cannot generate doses on logscale with minimum dose of 0. Use `include_zero=True` instead."
                )
            logDmin = np.log10(Dmin)
            logDmax = np.log10(Dmax)
            d = np.logspace(logDmin, logDmax, num=n)
        else:
            d = np.linspace(Dmin, Dmax, num=n)
        if include_zero and Dmin > 0:
            d = np.append(0, d)
        doses.append(d)
    dosegrid = np.meshgrid(*doses)

    total_length = np.prod(npoints)
    return_d = np.zeros((total_length, len(dmin)))
    for i in range(return_d.shape[1]):
        return_d[:, i] = dosegrid[i].flatten()

    return np.vstack([return_d] * replicates)


def is_monotherapy_ND(d) -> bool:
    """Return True if no more than 1 drug is present in the given N-drug dose array.

    This should only be applied to a single row (i.e., a single dose combination).

    Note, this still returns True if no drugs are present.

    :param ArrayLike d: Dose array, shape (n_samples, n_drugs)
    :return bool: True if no more than 1 drug is present in the given N-drug dose array
    """
    vals, counts = np.unique(d > 0, return_counts=True)
    drugs_present_count = counts[vals]
    if len(drugs_present_count) == 0:
        return True
    return drugs_present_count[0] == 1


def get_monotherapy_mask_ND(d) -> Tuple[np.ndarray]:
    """Return a mask of rows where no more than 1 drug is present in the given N-drug dose array.

    This helps to set synergy to the default value for monotherapy combinations.

    :param ArrayLike d: Dose array, shape (n_samples, n_drugs)
    :return Tuple[np.ndarray]: Mask of rows where no more than 1 drug is present
    """
    return np.where(np.apply_along_axis(is_monotherapy_ND, 1, d))


def get_drug_alone_mask_ND(d, drug_idx: int) -> Tuple[np.ndarray]:
    """Return a mask of rows where only the requested drug is present.

    Note: other drugs are considered to be absent as long as they are at their minimum dose.

    :param ArrayLike d: Dose array, shape (n_samples, n_drugs)
    :param int drug_idx: Index of the drug to check for
    :return Tuple[np.ndarray]: Mask of rows where only the requested drug is present
    """
    return get_drug_subset_mask_ND(d, [drug_idx])


def get_drug_subset_mask_ND(d, drug_indices: Sequence[int]) -> Tuple[np.ndarray]:
    """Return a mask of rows where only the requested drugs are present.

    Note: other drugs are considered to be absent as long as they are at their minimum dose.

    :param ArrayLike d: Dose array, shape (n_samples, n_drugs)
    :param Sequence[int] drug_indices: Indices of the drugs to check for
    :return Tuple[np.ndarray]: Mask of rows where only the requested drugs are present
    """
    N = d.shape[1]
    mask = d[:, drug_indices[0]] >= 0  # This inits it to "True"
    for drug_idx in drug_indices:
        for other_idx in range(N):
            if other_idx == drug_idx:
                continue
            mask = mask & (d[:, other_idx] == np.min(d[:, other_idx]))
    return np.where(mask)


def is_on_grid(d) -> bool:
    """Return True if the doses are on a grid.

    Doses are on a grid if all possible combinations of unique doses are present.

    Parameters:
    -----------
    d
        Doses, shape (n_samples, n_drugs)
    """
    unique_doses = [np.unique(d[:, i]) for i in range(d.shape[1])]
    for unique_dose in product(*unique_doses):
        mask = np.where((d == unique_dose).all(axis=1))
        if len(mask[0]) == 0:  # unique dose not found
            return False
    return True


def aggregate_replicates(d, E, aggfunc=np.median):
    """Aggregate rows of d and E with repeated combination doses.

    Parameters
    ----------
    d
        Doses, shape (n_samples, n_drugs)
    E
        Responses, shape (n_samples,)
    aggfunc : Callable, optional
        Function to aggregate replicate values of E, default is np.median

    Returns
    -------
    d
        Unique doses, shape (n_unique_samples, n_drugs)
    E
        Aggregated responses, shape (n_unique_samples,)
    """
    d_unique, num_replicates = np.unique(d, axis=0, return_counts=True)
    if (num_replicates == 1).all():
        return d, E

    _LOGGER.info(f"Aggregating replicate doses using {aggfunc.__name__}")

    def _find_matching_rows(row, d):
        return np.where((d == row).all(axis=1))

    unique_dose_indices = [_find_matching_rows(unique_row, d) for unique_row in d_unique]
    E_agg = np.asarray([aggfunc(E[indices]) for indices in unique_dose_indices])

    return d_unique, E_agg

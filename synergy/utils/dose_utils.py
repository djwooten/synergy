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

import numpy as np


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

    If ```logscale``` is `True`, use ```include_zero``` instead of setting the min dose to `0`.

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


def make_dose_grid_multi(dmin, dmax, npoints, logscale=True, include_zero=False, replicates: int = 1):
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

    return np.hstack([return_d] * replicates)


def is_monotherapy_ND(d):
    """Return True if no more than 1 drug is present."""
    vals, counts = np.unique(d > 0, return_counts=True)
    drugs_present_count = counts[vals]
    if len(drugs_present_count) == 0:
        return True
    return drugs_present_count[0] == 1


def get_monotherapy_mask_ND(d):
    """Return a mask of where no more than 1 drug is present.

    This helps to set synergy to the default value for monotherapy combinations.
    """
    return np.where(np.apply_along_axis(is_monotherapy_ND, 1, d))


def get_drug_alone_mask_ND(d, drug_idx):
    """Find all dose combinations where only the requested drug is present.

    Note: other drugs are considered to be absent as long as they are at their minimum dose.
    """
    N = d.shape[1]
    mask = d[:, drug_idx] >= 0  # This inits it to "True"
    for other_idx in range(N):
        if other_idx == drug_idx:
            continue
        mask = mask & (d[:, other_idx] == np.min(d[:, other_idx]))
    return np.where(mask)


def get_num_replicates(d1, d2):
    """Given 1d dose arrays d1 and d2, determine how many replicates of each unique combination are present

    Parameters:
    -----------
    d1 : array_like, float
        Doses of drug 1

    d2 : array_like, float
        Doses of drug 2

    Returns:
    -----------
    replicates : numpy.array
        Counts of each unique dose combination
    """
    return np.unique(np.asarray([d1, d2]), axis=1, return_counts=True)[1]


@DeprecationWarning
def remove_replicates(d1, d2):
    """Given 1d dose arrays d1 and d2, remove replicates.

    This is needed sometimes for plotting, since some plot functions expect a single d1, d2 -> E for each dose.

    Parameters:
    -----------
    d1 : array_like, float
        Doses of drug 1

    d2 : array_like, float
        Doses of drug 2

    Returns:
    -----------
    d1 : array_like, float
        Doses of drug 1 without replicates

    d2 : array_like, float
        Doses of drug 2 without replicates
    """
    d = np.asarray(list(set(zip(d1, d2))))
    return d[:, 0], d[:, 1]

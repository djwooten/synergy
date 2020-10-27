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

def grid(d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=1, logscale=True, include_zero=False):
    replicates = int(replicates)

    if logscale:
        d1 = np.logspace(np.log10(d1min), np.log10(d1max), num=n_points1)
        d2 = np.logspace(np.log10(d2min), np.log10(d2max), num=n_points2)
    else:
        d1 = np.linspace(d1min, d1max, num=n_points1)
        d2 = np.linspace(d2min, d2max, num=n_points2)

    if include_zero and logscale:
        if d1min > 0:
            d1 = np.append(0, d1)
        if d2min > 0:
            d2 = np.append(0, d2)
    
    D1, D2 = np.meshgrid(d1,d2)
    D1 = D1.flatten()
    D2 = D2.flatten()

    D1 = np.hstack([D1,]*replicates)
    D2 = np.hstack([D2,]*replicates)

    return D1, D2

def grid_multi(dmin, dmax, npoints, logscale=True, include_zero=False):
    if not (len(dmin)==len(dmax) and len(dmin)==len(npoints)):
        return None
    doses = []
    for Dmin, Dmax, n in zip(dmin, dmax, npoints):
        if logscale:
            logDmin = np.log10(Dmin)
            logDmax = np.log10(Dmax)
            d = np.logspace(logDmin, logDmax, num=n)
        else:
            d = np.linspace(Dmin, Dmax, num=n)
        if include_zero and Dmin > 0:
            d = np.append(0, d)
        doses.append(d)
    dosegrid = np.meshgrid(*doses)
    
    if include_zero:
        total_length = np.prod([i+1 for i in npoints])
    else:
        total_length = np.prod(npoints)
    n = len(dmin)
    return_d = np.zeros((total_length, n))
    
    for i in range(n):
        return_d[:,i] = dosegrid[i].flatten()

    return return_d


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
    return np.unique(np.asarray([d1,d2]), axis=1, return_counts=True)[1]

def remove_replicates(d1, d2):
    """Given 1d dose arrays d1 and d2, remove replicates. This is needed sometimes for plotting, since some plot functions expect a single d1, d2 -> E for each dose.

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
    return d[:,0], d[:,1]
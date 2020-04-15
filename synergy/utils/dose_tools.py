"""
    Copyright (C) 2020 David J. Wooten

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

def grid(d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=1, logspace=True, include_zero=False):
    replicates = int(replicates)

    if logspace:
        d1 = np.logspace(np.log10(d1min), np.log10(d1max), num=n_points1)
        d2 = np.logspace(np.log10(d2min), np.log10(d2max), num=n_points2)
    else:
        d1 = np.linspace(d1min, d1max, num=n_points1)
        d2 = np.linspace(d2min, d2max, num=n_points2)

    if include_zero:
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
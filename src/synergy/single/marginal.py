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

class MarginalLinear:
    """Some drugs' dose response may not follow some known parametric equation. In these cases, MarginalLinear() fits a piecewise linear model mapping log(d) -> E
    """
    def __init__(self, d=None, E=None, aggregation_function=np.mean):
        """
        Parameters
        ----------
        d : array-like
            Array of doses measured
        
        E : array-like
            Array of effects measured at doses d

        aggregation_function : function, default=np.mean
            If d contains repeated values (e.g., experimental replicates using the same dose many times), E at these repeated doses will be averaged using aggregation_function(E[d==X])
        """

        if (len(d) > len(np.unique(d))):
            self._d = []
            self._E = []

            # Given repeated dose measurements, average E for each
            for val in np.unique(d):
                self._d.append(val)
                self._E.append(aggregation_function(E[d==val]))

            self._d = np.asarray(self._d)
            self._E = np.asarray(self._E)

        else:
            self._d = np.array(d,copy=True)
            self._E = np.array(E,copy=True)

        # Replace 0 doses with the minimum float value
        self._d[self._d==0] = np.nextafter(0,1)

        # Sort doses and E
        sorted_indices = np.argsort(self._d)
        self._d = self._d[sorted_indices]
        self._E = self._E[sorted_indices]
        
        # Get log-transformed dose (used for interpolation)
        self._logd = np.log(self._d)

    def fit(self, d, E, aggregation_function=np.mean):
        """Calls __init__(d, E, aggregation_function)

        Parameters
        ----------
        d : array-like
            Array of doses measured
        
        E : array-like
            Array of effects measured at doses d

        aggregation_function : function, default=np.mean
            If d contains repeated values (e.g., experimental replicates using the same dose many times), E at these repeated doses will be averaged using aggregation_function(E[d==X])
        """
        
        self.__init__(d=d, E=E)

    def E(self, d):
        """Evaluate this model at dose d

        Parameters
        ----------
        d : array-like
            Doses to calculate effect at
        
        Returns
        ----------
        effect : array-like
            Evaluate's the model at dose in d
        """
        d = np.array(d, copy=True)
        d[d==0] = np.nextafter(0,1)
        logd = np.log(d)

        # These doses are below the minimum dose, or above the maximum, and therefore cannot be used for interpolation
        invalid_mask = np.where((logd < min(self._logd)) | (logd > max(self._logd)))

        E = np.interp(logd, self._logd, self._E)

        E[invalid_mask] = np.nan

        return E

    def E_inv(self, E):
        """Find the dose that will achieve effects in E. Only works for dose responses with strictly increasing or strictly decreasing E.

        Parameters
        ----------
        E : array-like
            Effects to get the doses for
        
        Returns
        ----------
        doses : array-like
            Doses which achieve effects E using this model.
        """
        if not self._is_invertible():
            ret = 0*E
            ret[:]=np.nan
            return ret
        
        # These effects are below the minimum or above the maximum, and therefore cannot be used for interpolation
        invalid_mask = np.where((E < min(self._E)) | (E > max(self._E)))

        d = np.exp(np.interp(E, self._E, self._logd))

        d[invalid_mask] = np.nan

        return d

    def _is_invertible(self):
        """The dose model is only invertible if E is strictly increasing or strictly decreasing
        """
        
        if (len(self._E) > 1):
            consecutive_diffs = np.ediff1d(self._E)
            if np.all(consecutive_diffs>0) or np.all(consecutive_diffs<0):
                return True
        return False
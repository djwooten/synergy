#    Copyright (C) 2024 David J. Wooten
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

from synergy.exceptions import ModelNotParameterizedError
from synergy.single.dose_response_model_1d import DoseResponseModel1D


class LogLinear(DoseResponseModel1D):
    """A model that fits dose response curves as piecewise linear interpolations of E vs log(dose).

    This model is useful for drugs whose dose response do not follow some known parametric equation.
    """

    def __init__(self, aggregation_function=np.median, nan_inverses=False, **kwargs):
        """Ctor.

        Dose response curves are typically strictly monotonic, however due to measurement noise or intrinsic biology, a
        piecewise linear model of the dose-response curve may not be strictly monotonic (e.g., non-invertible). If
        nan_inverses==True, all E's that do not correspond to a unique d will return E_inv(E)=NaN. If
        nan_inverses=False, non-monotonic regions will be cut out and replaced with a monotonic, log-linear curve so
        that E_inv can be uniquely calculated.

        :param Callable aggregation_function: Used to compute a representative effect given replicate dose measurements
        :param bool nan_inverses: If True, the fit model will return np.nan for E_inv(E) where E is in a non-invertible
        """
        self._d = np.asarray([])
        self._E = np.asarray([])
        self._logd = np.asarray([])
        self._aggregation_function = aggregation_function
        self._nan_inverses = nan_inverses

        # These will be filled based on which regions are invertible
        self._logd_for_inverse = np.asarray([])  # This stores all the log(d)'s used to construct a monotonic inverse
        self._E_for_inverse = np.asarray([])  # Thisi stores all the E's used to construct a monotonic inverse
        self._uninvertible_domains = []
        self._ready_for_inverse = False

        self._dose_scale = 1.0

    @property
    def is_specified(self):
        return len(self._logd) > 0 and len(self._E) > 0

    @property
    def is_fit(self):
        return self.is_specified

    def fit(self, d, E, **kwargs):
        self._ready_for_inverse = False

        if len(d) > len(np.unique(d)):
            d_uniques = []
            E_represntatives = []

            # Given repeated dose measurements, average E for each
            for val in np.unique(d):
                d_uniques.append(val)
                E_represntatives.append(self._aggregation_function(E[d == val]))

            self._d = np.asarray(d_uniques)
            self._E = np.asarray(E_represntatives)

        else:
            self._d = np.array(d, copy=True)
            self._E = np.array(E, copy=True)

        # Replace d=0 with the minimum (positive) float value (required for log-linearization)
        # TODO: A better solution would be to linearly interpolate between d=0 and the next dose up
        self._d[self._d == 0] = np.nextafter(0, 1)

        # Sort doses and E
        sorted_indices = np.argsort(self._d)
        self._d = self._d[sorted_indices]
        self._E = self._E[sorted_indices]

        # Calculate dose scale
        self._dose_scale = np.exp(np.mean(np.log(self._d)))

        # Get log-transformed dose (used for interpolation)
        self._logd = np.log(self._d / self._dose_scale)

    def E(self, d):
        if not self.is_specified:
            raise ModelNotParameterizedError("Must call fit() before calling E().")

        d = np.array(d, copy=True, dtype=np.float64)
        d[d == 0] = np.nextafter(0, 1)
        logd = np.log(d / self._dose_scale)

        E = np.interp(logd, self._logd, self._E, left=np.nan, right=np.nan)

        return E

    def E_inv(self, E):
        if not self.is_specified:
            raise ModelNotParameterizedError("Must call fit() before calling E_inv().")

        if not self._ready_for_inverse:
            self._prepare_inverse()

        if len(self._logd_for_inverse) == 0:
            return E * np.nan

        # These effects are below the minimum or above the maximum, and therefore cannot be used for interpolation
        invalid_mask = (E < min(self._E)) | (E > max(self._E))

        if self._nan_inverses:
            # Any E inside a domain we have found to be un-invertible is also invalid. If _nan_inverses is False
            # (default), the entire un-invertible domain will just be filled with a straight line. In this region, the
            # model will have E_inv(E(d)) != d. If nan_inverses is True, this whole region is exlucded (np.nan)
            for domain in self._uninvertible_domains:
                a, b = sorted(domain)
                invalid_mask = invalid_mask | ((E > a) & (E < b))

        d = np.exp(np.interp(E, self._E_for_inverse, self._logd_for_inverse))

        if hasattr(E, "__iter__"):
            invalid_mask = np.where(invalid_mask)
            d[invalid_mask] = np.nan
        elif invalid_mask:
            d = np.nan

        return d * self._dose_scale

    @staticmethod
    def create_fit(d, E, aggregation_function=np.median):
        """Factory method to build a log-linear model directly from data."""
        drug = LogLinear(aggregation_function=aggregation_function)
        drug.fit(d, E)
        return drug

    def _prepare_inverse(self):
        """Find the uninvertable domains prior to calculating E_inv().

        This is not done in .fit() because it is not necessary for the forward calculation of E(d). But it is necessary
        for calls of E_inv(E), so it is called the first time E_inv() is called.
        """
        if not self.is_specified:
            raise ModelNotParameterizedError("Must run fit() before preparing for inverse.")

        self._get_uninvertible_domains(self._E)

        # Which data points fall inside which uninvertable domains?
        index_to_udomain = dict()

        # Which data points are outside of uninvertable domains?
        valid_indices = []

        # These are the points we will use for interpolation. We will later extend them by the boundaries of the
        # uninvertible domains
        valid_E = []
        valid_d = []

        # Loop over every point and check whether it is in a u_domain
        for i, E in enumerate(self._E):
            found = False
            for domain in self._uninvertible_domains:
                if min(domain) <= E <= max(domain):
                    index_to_udomain[i] = domain
                    found = True
                    break
            if not found:
                valid_indices.append(i)
                valid_E.append(E)
                valid_d.append(self._logd[i])

        # Imagine 6 data points [0, 1, 2, 3, 4, 5]
        # If valid_indices = [0, 1, 5] then bad_indices = [2, 3, 4]
        # 2 is adjacent to a valid index, so we interpolate between them
        # Likewise for 4
        # General strategy - loop over bad_indices, loop over +/-1 neighbors, if in valid_indices, find boundary point

        for invalid_i in index_to_udomain.keys():
            for neighbor in [-1, 1]:
                neighbor_i = invalid_i + neighbor
                if neighbor_i in valid_indices:
                    E = self._E[neighbor_i]
                    ld = self._logd[neighbor_i]

                    E2 = self._E[invalid_i]
                    ld2 = self._logd[invalid_i]
                    domain = index_to_udomain[invalid_i]

                    # Is the valid neigbor above or below the domain?
                    if E < min(domain):  # Below
                        E_target = min(domain)
                        Elower = E
                        Eupper = E2
                        dlower = ld
                        dupper = ld2
                    else:  # above
                        E_target = max(domain)
                        Elower = E2
                        Eupper = E
                        dlower = ld2
                        dupper = ld
                    logd_target = np.interp(E_target, [Elower, Eupper], [dlower, dupper])

                    # These are new points that exist along the log-linear interpolation, where (d, E) exist at the
                    # boundary of un-invertible domains
                    valid_d.append(logd_target)
                    valid_E.append(E_target)

        # What about uninvertible domains that don't touch, but have no good
        # point between them? E.g., E3 is in (0.5,0.6), and E4 is in (0.7,0.8)
        #  - then we need to create two new points - at the inner boundaries
        #    of both domains
        for _i in index_to_udomain.keys():
            dom_i = index_to_udomain[_i]

            _j = _i + 1
            if _j in index_to_udomain:
                dom_j = index_to_udomain[_j]
                if dom_j == dom_i:
                    continue  # They are in the same domain

                E_i = self._E[_i]  # This is inside dom_i
                ld_i = self._logd[_i]
                E_j = self._E[_j]  # This is inside dom_j
                ld_j = self._logd[_j]

                if E_i < E_j:
                    Elower = E_i
                    Eupper = E_j
                    dlower = ld_i
                    dupper = ld_j
                    E_boundary_i = max(dom_i)
                    E_boundary_j = min(dom_j)
                else:
                    Elower = E_j
                    Eupper = E_i
                    dlower = ld_j
                    dupper = ld_i
                    E_boundary_j = max(dom_j)
                    E_boundary_i = min(dom_i)

                ld_boundary_i = np.interp(E_boundary_i, [Elower, Eupper], [dlower, dupper])
                ld_boundary_j = np.interp(E_boundary_j, [Elower, Eupper], [dlower, dupper])

                valid_d.append(ld_boundary_i)
                valid_E.append(E_boundary_i)
                valid_d.append(ld_boundary_j)
                valid_E.append(E_boundary_j)

        sorted_indices = np.argsort(valid_E)

        self._E_for_inverse = np.asarray(valid_E)[sorted_indices]
        self._logd_for_inverse = np.asarray(valid_d)[sorted_indices]
        self._ready_for_inverse = True

    def _get_uninvertible_domains(self, E):
        """Determine which interpolations between E's overlap

        For example, given a dose response that looks like
        ```
        5  E
        4  |-
        3  |   -    -
        2  |     -      -
        1  |                 -
        0  +------------------d
        ```
        there is no way to uniquely invert the range 2 <= E <= 3
        """
        if not self.is_specified:
            raise ModelNotParameterizedError("Model must be fit before it can be inverted")

        self._uninvertible_domains = []
        for E_idx_1 in range(len(E) - 2):
            for E_idx_2 in range(E_idx_1 + 1, len(E) - 1):
                E_interval_1 = E[E_idx_1], E[E_idx_1 + 1]
                E_interval_2 = E[E_idx_2], E[E_idx_2 + 1]
                overlap = self._interval_intersection(E_interval_1, E_interval_2)
                if overlap:
                    # Does this overlap with any regions we have seen before?
                    previous_overlaps = []  # We will merge these all together
                    for prev_olap in self._uninvertible_domains:
                        intersection_with_prev_olap = self._interval_intersection(overlap, prev_olap, inclusive=True)
                        if intersection_with_prev_olap:
                            previous_overlaps.append(prev_olap)

                    if len(previous_overlaps) == 0:  # No previous overlap
                        self._uninvertible_domains.append(overlap)
                    else:  # Merge this set with all previous overlaps
                        for po in previous_overlaps:  # remove previous ones
                            self._uninvertible_domains.remove(po)
                        self._uninvertible_domains.append(self._interval_hull_multiple(previous_overlaps + [overlap]))

    def _interval_intersection(self, interval_1: tuple, interval_2: tuple, inclusive: bool = False) -> tuple:
        """Find the intersection between two intervals.

        Inputs like
        ```
        |--int_1--|
                     |--int_2--|
        ```
        will return an empty interval ().

        Inputs like
        ```
        |----int_1----|
           |------int_2-----|
           a          b
        ```
        will return the interval (a, b).

        Inputs like
        ```
        |--int_1--|
                  |--int_2--|
                  a
        ```
        will return the interval (a, a) only if inclusive == True, otherwise will return an empty interval.

        Interval order does not matter.

        :param tuple interval_1: A 2-tuple of floats
        :param tuple interval_2: A 2-tuple of floats
        :param bool inclusive: If True, will include the boundary points as a potential intersection
        """
        interval_1 = tuple(sorted(interval_1))  # (low, high)
        interval_2 = tuple(sorted(interval_2))  # (low, high)

        # If the low value of one interval is above the high value of the other, there can be no overlap
        if not inclusive:
            if interval_2[0] >= interval_1[1] or interval_1[0] >= interval_2[1]:
                return ()  # No overlap
        else:
            if interval_2[0] > interval_1[1] or interval_1[0] > interval_2[1]:
                return ()  # No overlap

        # The low value is the max of the two low values, and the high value is the min of the two high values
        return max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])

    def _interval_hull_multiple(self, intervals: list) -> tuple:
        """Find the convex hull of all the given intervals.
        Inputs like
        ```
        |--int_1--|
                     |--int_2--|
        a                      b
        ```
        will return the interval (a, b)
        """
        if len(intervals) == 0:
            return ()

        if len(intervals) == 1:
            return intervals[0]

        min_val, max_val = intervals[0]
        for interval in intervals[1:]:
            min_val = min(min_val, min(interval))
            max_val = max(max_val, max(interval))

        return min_val, max_val

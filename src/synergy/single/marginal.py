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

class MarginalLinear:
    """Some drugs' dose response may not follow some known parametric equation. In these cases, MarginalLinear() fits a piecewise linear model mapping log(d) -> E

    Parameters
    ----------
    aggregation_function : function, default=np.mean
        If d contains repeated values (e.g., experimental replicates using the same dose many times), E at these repeated doses will be averaged using aggregation_function(E[d==X])

    strict_inverse : bool, default=False
        Dose response curves are typically strictly monotonic, however due to measurement noise or intrinsic biology, a piecewise linear model of the dose-response curve may not be strictly monotonic (e.g., non-invertible). If strict_inverse==True, all E's that do not correspond to a unique d will return E_inv(E)=NaN. If strict_inverse=False, non-monotonic regions will be cut out and replaced with a monotonic, log-linear curve so that E_inv can be uniquely calculated.
    """

    # NOTE on __init__() and fit(): **kwargs is only present to avoid errors
    # people may run into when reusing code for fitting Hill functions in which
    # they set kwargs. There are no kwargs used for these methods
    def __init__(self, aggregation_function=np.mean, strict_inverse=False, **kwargs):
        self._d = None
        self._E = None
        self._logd = None
        self._aggregation_function = aggregation_function
        self._fit = False
        self._strict_inverse = strict_inverse

        self.E0 = None
        self.Emax = None
        
        
        # These will be filled based on which regions are invertible
        self._logd_for_inverse = None
        self._E_for_inverse = None
        self._uninvertible_domains = []
        self._ready_for_inverse = False

    def fit(self, d, E, **kwargs):
        """Calls __init__(d, E, aggregation_function)

        Parameters
        ----------
        d : array_like
            Array of doses measured
        
        E : array_like
            Array of effects measured at doses d
        """
        self._fit  = True
        self._ready_for_inverse = False

        if (len(d) > len(np.unique(d))):
            self._d = []
            self._E = []

            # Given repeated dose measurements, average E for each
            for val in np.unique(d):
                self._d.append(val)
                self._E.append(self._aggregation_function(E[d==val]))

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

        self.E0 = self._E[0]
        self.Emax = self._E[-1]
        
        # Get log-transformed dose (used for interpolation)
        self._logd = np.log(self._d)

    def E(self, d):
        """Evaluate this model at dose d

        Parameters
        ----------
        d : array_like
            Doses to calculate effect at
        
        Returns
        ----------
        effect : array_like
            Evaluate's the model at dose in d
        """
        if not self._fit: return d*np.nan

        d = np.array(d, copy=True)
        d[d==0] = np.nextafter(0,1)
        logd = np.log(d)

        E = np.interp(logd, self._logd, self._E, left=np.nan, right=np.nan)

        return E

    def E_inv(self, E):
        """Find the dose that will achieve effects in E. Only works for dose responses with strictly increasing or strictly decreasing E.

        Parameters
        ----------
        E : array_like
            Effects to get the doses for
        
        Returns
        ----------
        doses : array_like
            Doses which achieve effects E using this model.
        """
        if not self._fit: return E*np.nan
        if not self._ready_for_inverse: self._prepare_inverse()
        if len(self._logd_for_inverse)==0: return E*np.nan

        # These effects are below the minimum or above the maximum, and therefore cannot be used for interpolation
        invalid_mask = (E < min(self._E)) | (E > max(self._E))

        
        if self._strict_inverse:
            # Any E inside a domain we have found to be un-invertible is also invalid. If _strict_inverse is False (default), the entire un-invertible domain will just be filled with a straight line. In this region, the model would have E_inv(E(d))!=d. If strict_inverse is True, this whole region is exlucded (np.nan)
            for domain in self._uninvertible_domains:
                a,b = sorted(domain)
                invalid_mask = invalid_mask | ((E > a) & (E < b))
        
        d = np.exp(np.interp(E, self._E_for_inverse, self._logd_for_inverse))

        if hasattr(E, "__iter__"):
            invalid_mask = np.where(invalid_mask)
            d[invalid_mask] = np.nan
        elif invalid_mask:
            d = np.nan

        return d

    def create_fit(d, E, aggregation_function=np.mean):
        """Courtesy function to build a marginal linear model directly from 
        data. Initializes a model using the provided aggregation function, then 
        fits.
        """
        drug = MarginalLinear(aggregation_function=aggregation_function)
        drug.fit(d, E)
        return drug

    def is_fit(self):
        return self._fit


    def _prepare_inverse(self):
        if not self._fit: return
        self._get_invertible_domains(self._E)
        
        # Which data points fall inside which uninvertable domains?
        index_to_udomain = dict()

        # Which data points are outside of uninvertable domains?
        valid_indices = []

        # These are the points we will use for interpolation. We will later extend them by the boundaries of the uninvertible domains
        valid_E = []
        valid_d = []

        # Loop over every point and check whether it is in a u_domain
        for i,E in enumerate(self._E):
            found = False
            for domain in self._uninvertible_domains:
                if E >= min(domain) and E <= max(domain):
                    index_to_udomain[i] = domain
                    found = True
                    break
            if not found:
                valid_indices.append(i)
                valid_E.append(E)
                valid_d.append(self._logd[i])

        # Imagine 6 data points [0, 1, 2, 3, 4, 5]
        # If valid_indices = [0,1,5] then bad_indices = [2, 3, 4]
        # 2 is adjacent to a valid index, so we interpolate between them
        # Likewise for 4
        # General strategy - loop over bad_indices, loop over +/-1 neighbors, if in valid_indices, find boundary point

        for invalid_i in index_to_udomain.keys():
            for neighbor in [-1,1]:
                neighbor_i = invalid_i + neighbor
                if neighbor_i in valid_indices:
                    E = self._E[neighbor_i]
                    ld = self._logd[neighbor_i]

                    E2 = self._E[invalid_i]
                    ld2 = self._logd[invalid_i]
                    domain = index_to_udomain[invalid_i]


                    # Is the valid neigbor above or below the domain?
                    if E < min(domain): # Below
                        E_target = min(domain)
                        Elower = E
                        Eupper = E2
                        dlower = ld
                        dupper = ld2
                    else: # above
                        E_target = max(domain)
                        Elower = E2
                        Eupper = E
                        dlower = ld2
                        dupper = ld
                    logd_target = np.interp(E_target, [Elower, Eupper], [dlower, dupper])

                    # These are new points that exist along the log-linear interpolation, where (d,E) exist at the boundary of un-invertible domains
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
                if dom_j == dom_i: continue # They are in the same domain

                E_i = self._E[_i] # This is inside dom_i
                ld_i = self._logd[_i]
                E_j = self._E[_j] # This is inside dom_j
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

    def _get_invertible_domains(self, E):
        if not self._fit: return
        self._uninvertible_domains = []
        for i in range(len(E)-2):
            for j in range(i+1, len(E)-1):
                r1 = [E[i], E[i+1]]
                r2 = [E[j], E[j+1]]
                overlap = self._interval_intersection(r2,r1)
                if overlap is not None:
                    # Does this overlap with any regions we have seen before?
                    previous_overlaps = [] # We will merge these all together
                    for _i, prev_olap in enumerate(self._uninvertible_domains):
                        overlap_2 = self._interval_intersection(overlap, prev_olap, border_is_overlap=True)
                        if overlap_2 is not None:
                            previous_overlaps.append(prev_olap)

                    if len(previous_overlaps)==0: # No previous overlap
                        self._uninvertible_domains.append(overlap)
                    else: # Merge this set with all previous overlaps
                        for po in previous_overlaps: # remove previous ones
                            self._uninvertible_domains.remove(po)
                        self._uninvertible_domains.append(self._interval_union_multiple(previous_overlaps + [overlap,]))

    def _interval_intersection(self, r1, r2, border_is_overlap=False):
        r1 = sorted(r1) # low -> high
        r2 = sorted(r2) # low -> high
        
        # If the low value of one interval is above the high value of the other, there can be no overlap
        if not border_is_overlap:
            if (r2[0] >= r1[1] or r1[0] >= r2[1]): return None
        else:
            if (r2[0] > r1[1] or r1[0] > r2[1]): return None

        # The low value is the max of the two low values, and the high value is the min of the two high values
        return (max(r1[0],r2[0]), min(r1[1],r2[1]))

    def _interval_union(self, r1, r2):
        r1 = sorted(r1)
        r2 = sorted(r2)
        return (min(r1[0], r2[0]), max(r1[1], r2[1]))

    def _interval_union_multiple(self, sets):
        if len(sets)==0: return None
        if len(sets)==1: return sets[0]
        r1 = sets[0]
        for r2 in sets[1:]:
            r1 = self._interval_union(r1, r2)
        return r1
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


def sham(d, drug):
    """Simulates a sham combination experiment. In a sham experiment, the two drugs combined are (secretly) the same drug. For example, a sham combination may add 10uM drugA + 20uM drugB. But because drugA and drugB are the same (drugX), the combination is really just equivalent to 30uM of the drug.

    
    """
    if not 0 in d:
        d = np.append(0,d)
    d1, d2 = np.meshgrid(d,d)
    d1 = d1.flatten()
    d2 = d2.flatten()
    E = drug.E(d1+d2)
    return d1, d2, E

def remove_zeros(d):
    d=np.array(d,copy=True)
    dmin = np.min(d[d>0]) # smallest nonzero dose
    dmax = np.max(d) # Largest dose
    dilution = dmin/dmax # What is the total dose range tested (in logspace)
    d[d==0]=dmin * dilution**10
    #d[d==0] = np.nextafter(0,1) # Smallest positive float
    return d

def remove_zeros_onestep(d):
    d=np.array(d,copy=True)
    dmin = np.min(d[d>0]) # smallest nonzero dose
    dmin2 = np.min(d[d>dmin])
    dilution = dmin/dmin2
    d[d==0]=dmin * dilution
    return d

def residual_ss(d1, d2, E, model):
    E_model = model(d1, d2)
    return np.sum((E-E_model)**2)

def AIC(sum_of_squares_residuals, n_parameters, n_samples):
    """
    SOURCE: AIC under the Framework of Least Squares Estimation, HT Banks, Michele L Joyner, 2017
    Equations (6) and (16)
    https://projects.ncsu.edu/crsc/reports/ftp/pdf/crsc-tr17-09.pdf
    """
    aic = n_samples * np.log(sum_of_squares_residuals / n_samples) + 2*(n_parameters + 1)
    if n_samples / n_parameters > 40:
        return aic
    else:
        return aic + 2*n_parameters*(n_parameters+1) / (n_samples - n_parameters - 1)

def BIC(sum_of_squares_residuals, n_parameters, n_samples):
    return n_samples * np.log(sum_of_squares_residuals / n_samples) + (n_parameters+1)*np.log(n_samples)

def r_squared(E, sum_of_squares_residuals):
    ss_tot = np.sum((E-np.mean(E))**2)
    return 1-sum_of_squares_residuals/ss_tot

def sanitize_initial_guess(p0, bounds):
    """
    Makes sure p0 is within the bounds
    """
    index = 0
    for x, lower, upper in zip(p0, *bounds):
        if x is None:
            if True in np.isinf((lower,upper)): np.min((np.max((0,lower)), upper))
            else: p0[index]=np.mean((lower,upper))

        elif x < lower: p0[index]=lower
        elif x > upper: p0[index]=upper
        index += 1

   

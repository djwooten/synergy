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
import inspect
import warnings


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

def sham_higher(d, drug, n_drugs):
    """Simulates a sham combination experiment for 3+ drugs. In a sham experiment, the two drugs combined are (secretly) the same drug. For example, a sham combination may add 10uM drugA + 20uM drugB. But because drugA and drugB are the same (drugX), the combination is really just equivalent to 30uM of the drug.

    Parameters
    ----------
    d : array_like
        Dose escalation to use for each "sham" drug

    drug
        A parameterized drug model from synergy.single, such as Hill, Hill_2P, Hill_CI, or MarginalLinear.

    n_drugs : int
        The number of drugs to include in the sham combination

    Returns
    ----------
    doses : M x n_drugs numpy.ndarray
        All dose pairs for each combination. In total, there are M samples, taken from n_drugs drugs.

    E : numpy.array
        Sham effects calculated as drug.E(doses.sum(axis=1))
    """
    if not 0 in d:
        d = np.append(0,d)
    doses = [d, ]*n_drugs
    doses = list(np.meshgrid(*doses))
    for i in range(n_drugs):
        doses[i] = doses[i].flatten()
    doses = np.asarray(doses).T
    E = drug.E(doses.sum(axis=1))
    return doses, E

def remove_zeros(d, min_buffer=0.2):
    """Replace zeros with some semi-intelligently chosen small value

    When plotting on a log scale, 0 doses can cause problems. This replaces all 0's using the dilution factor between the smallest non-zero, and second-smallest non-zero doses. If that dilution factor is too close to 1, it will replace 0's doses with a dose that is min_buffer*(max(d)-min(d[d>0])) less than min(d[d>0]) on a log scale.

    Parameters
    ----------
    d : array_like
        Doses to remove zeros from. Original array will not be changed.

    min_buffer : float , default=0.2
        For very large dose arrays with very small step sizes (useful for getting smooth plots), replacing 0's may lead to a value too close to the smallest non-zero dose. min_buffer is the minimum buffer (in log scale, relative to the full dose range) that 0's will be replaced with.
    """

    d=np.array(d,copy=True)
    dmin = np.min(d[d>0]) # smallest nonzero dose
    dmin2 = np.min(d[d>dmin])
    dilution = dmin/dmin2

    dmax = np.max(d)
    logdmin = np.log(dmin)
    logdmin2 = np.log(dmin2)
    logdmax = np.log(dmax)

    if (logdmin2-logdmin) / (logdmax-logdmin) < min_buffer:
        logdmin2_effective = logdmin + min_buffer*(logdmax-logdmin)
        dilution = dmin/np.exp(logdmin2_effective)

    d[d==0]=dmin * dilution
    return d

def residual_ss(d1, d2, E, model):
    E_model = model(d1, d2)
    return np.sum((E-E_model)**2)

def residual_ss_1d(d, E, model):
    E_model = model(d)
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

def sanitize_single_drug_model(model, default_class, expected_superclass=None, **kwargs):
    """
    Makes sure the given single drug model is a class or object of a class that is permissible for the given synergy model.

    Parameters
    ----------
    model : object or class
        A single drug model

    default_class : class
        The type of model to return if the given model is of the wrong type

    expected_superclass : class , default=None
        The class the model is expected to be an instance of

    Returns
    -------
    model : object
        An object that is an instance of expected_superclass
    """
    # The model is a class
    if inspect.isclass(model):

        # If there is no expected_superclass, assume the given class is fine
        if expected_superclass is None:
            return model(**kwargs)

        else:
            # The model is a subclass of the expected subclass
            if issubclass(model, expected_superclass):
                # We are good!
                return model(**kwargs)

            # The given class violates the expected class: return the default
            else:
                if model is not None: warnings.warn("Expected single drug model to be subclass of %s, instead got %s"%(expected_superclass, model))
                return default_class(**kwargs)
        
        return model(**kwargs)

    # The model is an object
    else:
        # There is no expected_superclass, so assume the object is fine
        if expected_superclass is None:
            if model is None:
                return default_class(**kwargs)
            return model
        
        # The model is an instance of the expected_superclass, so good!
        elif isinstance(model,expected_superclass):
            return model

        # The model is an instance of the wrong type of class, so return the default
        else:
            if model is not None: warnings.warn("Expected single drug model to be subclass of %s, instead got %s"%(expected_superclass, type(model)))

            return default_class(**kwargs)
    return default_class(**kwargs)

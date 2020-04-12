import numpy as np
import synergy.single.hill as hill


def fit_single(d, E, E0_bounds, Emax_bounds, h_bounds, C_bounds):
    drug = hill.Hill(E0_bounds=E0_bounds, Emax_bounds=Emax_bounds, h_bounds=h_bounds, C_bounds=C_bounds)
    drug.fit(d, E)
    return drug

def fit_single_2parameter(d, E, h_bounds, C_bounds, E0=1., Emax=0.):
    drug = hill.Hill(h_bounds=h_bounds, C_bounds=C_bounds, E0=E0, Emax=Emax)
    drug.fit_2parameter(d, E)
    return drug

def sham(d, E0, Emax, h, C):
    drug = hill.Hill(E0=E0, Emax=Emax, h=h, C=C)
    if not 0 in d:
        d = np.append(d,0)
    d1, d2 = np.meshgrid(d,d)
    d1 = d1.flatten()
    d2 = d2.flatten()
    E = drug.E(d1+d2)
    return d1, d2, E

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
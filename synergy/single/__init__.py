import numpy as np

from synergy.single.hill import Hill, Hill_2P, Hill_CI
from synergy.single.nonparametric import LogLinear


def fit_hill(d, E, E0_bounds=None, Emax_bounds=None, h_bounds=None, C_bounds=None, **kwargs) -> Hill:
    """Build a Hill model, fit it to data, and return it."""
    drug = Hill(E0_bounds=E0_bounds, Emax_bounds=Emax_bounds, h_bounds=h_bounds, C_bounds=C_bounds)
    drug.fit(d, E, **kwargs)
    return drug


def fit_hill_2p(d, E, E0=1, Emax=0, h_bounds=None, C_bounds=None, **kwargs) -> Hill_2P:
    """Build a 2-parameter Hill model, fit it to data, and return it."""
    drug = Hill_2P(E0=E0, Emax=Emax, h_bounds=h_bounds, C_bounds=C_bounds)
    drug.fit(d, E, **kwargs)
    return drug


def fit_hill_CI(d, E) -> Hill_CI:
    """Build a Combination Index model, fit it to data, and return it."""
    drug = Hill_CI()
    drug.fit(d, E)
    return drug


def fit_loglinear(d, E, aggregation_function=np.median) -> LogLinear:
    """Build a log-linear dose response model, fit it to data, and return it."""
    drug = LogLinear(aggregation_function=aggregation_function)
    drug.fit(d, E)
    return drug

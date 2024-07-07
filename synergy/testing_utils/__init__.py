import numpy as np


def unique_tol(a, tol=1e-8):
    """Return the unique elements of a numpy array, with a tolerance.

    Parameters:
    -----------
    a : array_like
        Input array
    tol : float, optional
        Tolerance for uniqueness, default is 1e-8

    Returns:
    --------
    unique : array_like
        Unique elements of a
    """
    return a[~(np.triu(np.abs(a[:, None] - a) <= tol, 1)).any(0)]

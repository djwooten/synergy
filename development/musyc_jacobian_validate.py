import numpy as np
from scipy.optimize import approx_fprime, check_grad

import synergy.combination.musyc as musyc
import synergy.combination.musyc_jacobian as musyc_jacobian

# Using jacobian slighly slows the fit when there are ~10-100 datapoints
# Using jacobian speeds the fit when there are > ~1000 datapoints
# Using jacobian improves convergence

E0, E1, E2, E3 = 1, 0.7, 0.5, 0
h1, h2 = 1.5, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 3.2, 0.1

model = musyc.MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

r1 = model.r1
r2 = model.r2

d = np.logspace(-3, 1, num=30)

# Numerically check gradient
logh1 = np.log(h1)
logC1 = np.log(C1)
logh2 = np.log(h2)
logC2 = np.log(C2)
logalpha12 = np.log(alpha12)
logalpha21 = np.log(alpha21)

# Jacobian return order
# E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21
f = lambda logalpha21: model._model(
    C1 * 2.0,
    C2 / 2.0,
    E0,
    E1,
    E2,
    E3,
    np.exp(logh1),
    np.exp(logh2),
    np.exp(logC1),
    np.exp(logC2),
    r1,
    r2,
    np.exp(logalpha12),
    np.exp(logalpha21),
)

fj = lambda i: approx_fprime(np.asarray([i]), f, 0.0001)
print(fj(logalpha21))

jac = lambda E0: musyc_jacobian.jacobian(
    C1 * 2.0, C2 / 2.0, E0, E1, E2, E3, logh1, logh2, logC1, logC2, r1, r2, logalpha12, logalpha21
)

#       jac                                 fj
#  2.23939104e-01, #    E0              0.2239391 # These are all correct
#  2.63567871e-01, #    E1              0.26356787 # Also, it makes sense they +
#  1.30952694e-01, #    E2              0.13095269
#  3.81540331e-01, #    E3              0.38154033
#  1.76168986e-02,      logh1           -0.12307921 # These could be + or (-)
# -2.82330684e-02,      logh2           -0.04041386 # - makes tentative sense
# -7.19772057e-02,      logC1           0.18284021 # These SHOULD be positive
#  4.10819668e-01,      logC2           0.08579638
#  7.54493621e-02,      logalpha12      -0.08728012 # These SHOULD be negative
#  3.00830660e-04])     logalpha21      -0.00333315

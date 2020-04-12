import numpy as np
import synergy.combination.musyc as musyc
import synergy.combination.musyc_jacobian as musyc_jacobian
from scipy.optimize import check_grad, approx_fprime

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

d = np.logspace(-3,1,num=30)

# Numerically check gradient
logh1 = np.log10(h1)
logC1 = np.log10(C1)
logh2 = np.log10(h2)
logC2 = np.log10(C2)
logalpha12 = np.log10(alpha12)
logalpha21 = np.log10(alpha21)

# Jacobian return order
# E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21
f = lambda logalpha12: model._model(C1*2., C2/2., E0, E1, E2, E3, 10.**logh1, 10.**logh2, 10.**logC1, 10.**logC2, r1, r2, 10.**logalpha12, 10.**logalpha21)

jac =  lambda logalpha12: musyc_jacobian.jacobian(C1*2., C2/2., E0, E1, E2, E3, 10.**logh1, 10.**logh2, 10.**logC1, 10.**logC2, r1, r2, 10.**logalpha12, 10.**logalpha21)

check_grad(f, jac, np.asarray([-2]))

fj = lambda i: approx_fprime(np.asarray([i]), f, 0.0001)
npfv = np.vectorize(fj)
npfv(np.asarray([-1,0,1]))

jac(np.asarray([-1,0,1]))

# npfv and jac should return the same values - CHECK musyc_jacobian - especially for alpha...
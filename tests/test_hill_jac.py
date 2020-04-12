import numpy as np
from synergy.single.hill import Hill
from scipy.optimize import check_grad, approx_fprime

# Using jacobian slighly slows the fit when there are ~10-100 datapoints
# Using jacobian speeds the fit when there are > ~1000 datapoints
# Using jacobian improves convergence

E0 = 1
Emax = 0
h = 2.3
C = 1e-2

model = Hill(E0=E0, Emax=Emax, h=h, C=C)

d = np.logspace(-3,0,num=20)
E = model.E(d) * (1+(np.random.rand(len(d))-0.5)/10)

print("Using jacobian")
#%timeit model.fit(d, E, use_jacobian=True, p0=[1,1,2,2])
# No noise
# 841 µs ± 2.83 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# With noise
# 848 µs ± 35 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# -- After changing from .transpose() to .hstack(.reshape(-1,1))
# 681 µs ± 6.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 957 µs ± 28.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# 2000 dose points
# 4.18 ms ± 16.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Poor initial guess, 20 points
# (1,1,1,1)
# 1.32 ms ± 12.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# (1,1,10,10)
# 6.47 ms ± 311 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Poor initial guess, 20 points, parralel (SLOOOOW)
# (1,1,2,2)
# 145 ms ± 3.71 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#    ^^^ ( Compare to 1.17 ms ± 11.7 µs using sequential)

print("Not using")
#%timeit model.fit(d, E, use_jacobian=False, p0=[1,1,2,2])
# No noise
# 773 µs ± 18.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# But sometimes gave a "Covariance could not be estimated warning"

# With noise
# 702 µs ± 5.66 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 585 µs ± 17.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 785 µs ± 17.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# 2000 dose points
# 6.17 ms ± 79.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Poor initial guess, 20 points
# (1,1,1,1)
# 1.29 ms ± 44.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# (1,1,10,10)
# Failed

# Poor initial guess, 20 points, parralel
# (1,1,2,2)
# 1.07 ms ± 8.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)




# Numerically check gradient
logh = np.log(h)
logC = np.log(C)
f = lambda logC: model._model(C*3, E0, Emax, np.exp(logh), np.exp(logC))
jac =  lambda logC: model._model_jacobian(C*3, E0, Emax, logh, logC)[:,3]

check_grad(f, jac, np.asarray([0]))

fj = lambda i: approx_fprime(np.asarray([i]), f, 0.0001)
npfv = np.vectorize(fj)
npfv(np.asarray([-1,0,1]))

jac(np.asarray([-1,0,1]))

# npfv and jac should return the same values - CHECK musyc_jacobian - especially for alpha...
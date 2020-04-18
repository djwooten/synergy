from matplotlib import pyplot as plt
import numpy as np
from synergy.combination import MuSyCGNew
from synergy.utils.dose_tools import grid

E0, E1, E2, E3 = 1, 0.6, 0.4, 0.
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
oalpha12, oalpha21 = 2., 1.
gamma12, gamma21 = 0.8, 1.3

alpha12 = MuSyCGNew._prime_to_alpha(oalpha12, C2, gamma12)
alpha21 = MuSyCGNew._prime_to_alpha(oalpha21, C1, gamma21)

truemodel = MuSyCGNew(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, oalpha12=oalpha12, oalpha21=oalpha21, gamma12=gamma12, gamma21=gamma21)


npoints = 8
npoints2 = 12
d1, d2 = grid(C1/1e3, C1*1e3, C2/1e3, C2*1e3, npoints, npoints2)

E = truemodel.E(d1, d2)

noise = 0.05
#E_fit = E*(1+(np.random.rand(len(E))-0.5)/3)
E_fit = E + noise*(E0-E3)*(2*np.random.rand(len(E))-1)

model = MuSyCGNew()
model.fit(d1, d2, E_fit, bootstrap_iterations=100)

print("\n")
print(model)
print(model.get_parameter_range().T)


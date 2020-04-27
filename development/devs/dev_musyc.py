import numpy as np
import pandas as pd

from synergy.combination import MuSyC
from synergy.utils.dose_tools import grid

E0, E1, E2, E3 = 1, 0.6, 0.4, 0.
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 2., 1.
gamma12, gamma21 = 1.3, 0.5

truemodel = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)


npoints = 8
npoints2 = 8
d1, d2 = grid(C1/1e2, C1*1e2, C2/1e2, C2*1e2, npoints, npoints2, include_zero=True)

E = truemodel.E(d1, d2)

noise = 0.05
E_fit = E + noise*(E0-E3)*(2*np.random.rand(len(E))-1)

model = MuSyC.create_fit(d1, d2, E_fit, bootstrap_iterations=10)

print("\n")
print(model)
if model.converged:
    print(model.get_parameter_range().T)


scatter_points = pd.DataFrame({'drug1.conc':d1, 'drug2.conc':d2, 'effect':E_fit})
DD1, DD2 = grid(C1/1e4, C1*1e4, C2/1e4, C2*1e4, npoints*2, npoints2*2)

model.plot_surface_plotly(DD1, DD2, fname="musyc_fit_new.html", scatter_points=scatter_points)

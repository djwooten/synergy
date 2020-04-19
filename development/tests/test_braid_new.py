import numpy as np
from synergy.combination import BRAID
from synergy.utils.dose_tools import grid
import pandas as pd

E0, E1, E2, E3 = 1, 0.5, 0.3, 0.
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
kappa = 0.8 # > 0
delta = 0.5 # < 1

npoints = 10

truemodel = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=kappa, delta=delta)

D1, D2 = grid(C1/1e2, C1*1e2, C2/1e2, C2*1e2, npoints, npoints)
E = truemodel.E(D1, D2)



Efit = E*(1+(np.random.rand(len(D1))-0.5)/4.)

model = BRAID()
model.fit(D1, D2, Efit, bootstrap_iterations=100)
print(model)
print(model.get_parameter_range().T)


scatter_points = pd.DataFrame({'drug1.conc':D1, 'drug2.conc':D2, 'effect':Efit})
DD1, DD2 = grid(C1/1e4, C1*1e4, C2/1e4, C2*1e4, npoints*2, npoints*2)

model.plot_surface_plotly(DD1, DD2, fname="braid_fit.html", scatter_points=scatter_points)

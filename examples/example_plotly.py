from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from synergy.combination import MuSyC
from synergy.utils import plots
from synergy.utils import dose_tools


E0, E1, E2, E3 = 1, 0.7, 0.6, 0.4
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 10.2, 1.1
gamma12, gamma21 = 1,1

model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)


replicates = 2
npoints = 6
npoints2 = 8
D1, D2 = dose_tools.grid(1e-3, 1, 1e-2, 10, npoints, npoints2, replicates=replicates, include_zero=True)

D1 = D1/3.

E = model.E(D1, D2)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/5.)

scatter_points = pd.DataFrame({'drug1.conc':D1, 'drug2.conc':D2, 'effect':Efit})
model.fit(D1, D2, Efit)


D1, D2 = dose_tools.grid(1e-3, 1, 1e-2, 10, npoints*10, npoints2*10, include_zero=True)
D1 = D1/3

model.plot_surface_plotly(D1, D2, scatter_points=scatter_points, xlabel="Drug1", ylabel="Drug2", zlabel="Effect", fname="plotly_musyc.html")
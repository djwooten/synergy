import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from synergy.utils import dose_tools, plots
from synergy.higher import MuSyC, HSA

# E_params = [1,0.5,0.5,0.5,0.5,0,0,0]
E_params = [1, 0.5, 0.5, 0.25, 0.5, 0.25, 0.25, 0.5**3]
h_params = [2, 1, 0.8]
C_params = [0.1, 0.01, 0.1]
alpha_params = [2, 3, 1, 1, 0.7, 0.5, 2, 1, 1]
# gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]
# alpha_params = [0,]*9
gamma_params = [
    1,
] * 9

params = E_params + h_params + C_params + alpha_params + gamma_params

truemodel = MuSyC()
truemodel.parameters = params

d = dose_tools.make_dose_grid_multi((1e-3, 1e-3, 1e-3), (1, 1, 1), (6, 6, 6), include_zero=True)

d1 = d[:, 0]
d2 = d[:, 1]
d3 = d[:, 2]
E = truemodel.E(d)

model = HSA()
model.fit(d, E)

model.plotly_isosurfaces()

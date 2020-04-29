import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from synergy.utils import dose_tools, plots
from synergy.higher import MuSyC, Bliss
from synergy.single import Hill

#E_params = [1,0.5,0.5,0.5,0.5,0,0,0]
E_params = [1,0.5,0.5,0.25,0.5,0.25,0.25,0.5**3]
h_params = [2,1,0.8]
C_params = [0.1,0.01,0.1]
alpha_params = [2,3,1,1,0.7,0.5,2,1,1]
#gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]
#alpha_params = [1,]*9
gamma_params = [1,]*9

params = E_params + h_params + C_params + alpha_params + gamma_params

truemodel = MuSyC()
truemodel.parameters = params

single_models = [Hill(E0=1, Emax=0.5, h=2, C=0.1), Hill(E0=1, Emax=0.5, h=1, C=0.01), Hill(E0=1, Emax=0.5, h=0.8, C=0.1)]
d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)

d1 = d[:,0]
d2 = d[:,1]
d3 = d[:,2]
E = truemodel.E(d)

model = Bliss(h_bounds=(1e-3,1e3))
model.fit(d, E, single_models=single_models)

model.plotly_isosurfaces(d)
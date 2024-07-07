import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from synergy.higher import MuSyC
from synergy.utils import dose_tools, plots

E_params = [2, 1, 1, 1, 1, 0, 0, 0]
h_params = [2, 1, 0.8]
C_params = [0.1, 0.01, 0.1]
alpha_params = [2, 3, 1, 1, 0.7, 0.5, 2, 1, 1]
gamma_params = [0.4, 2, 1, 2, 0.7, 3, 2, 0.5, 2]
# alpha_params = [1,]*9
# gamma_params = [1,]*9

params = E_params + h_params + C_params + alpha_params + gamma_params

truemodel = MuSyC()
truemodel.parameters = params


d = dose_tools.make_dose_grid_multi((1e-3, 1e-3, 1e-3), (1, 1, 1), (6, 6, 6), include_zero=True)
# truemodel.plotly_isosurfaces(d, vmin=0, vmax=2, isomin=0.2, isomax=2)

d1 = d[:, 0]
d2 = d[:, 1]
d3 = d[:, 2]
E = truemodel.E(d)
noise = 0.01
E_fit = E + noise * (E_params[0] - E_params[-1]) * (2 * np.random.rand(len(E)) - 1)

if False:
    fig = plt.figure(figsize=(15, 8))
    for i, DD in enumerate(np.unique(d3)):
        mask = np.where(d3 == DD)
        ax = fig.add_subplot(3, 4, i + 1)
        plots.plot_heatmap(d1[mask], d2[mask], E[mask], ax=ax, vmin=0, vmax=2)

    plt.tight_layout()
    plt.show()


model = MuSyC(E_bounds=(0, 2), h_bounds=(1e-3, 1e3), alpha_bounds=(1e-5, 1e5), gamma_bounds=(1e-5, 1e5))
model.fit(d, E_fit, bootstrap_iterations=20)

pars = model.get_parameters()

print(model.summary())

from matplotlib import pyplot as plt
import numpy as np
from synergy.combination import MuSyC, MuSyCG
from synergy.utils import plots
from synergy.utils.dose_tools import grid


E0, E1, E2, E3 = 1, 0.5, 0.3, 0.
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 1., 1.
gamma12, gamma21 = 0.4, 0.4

model = MuSyCG(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)

truemodel = MuSyCG(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)

model_no_gamma = MuSyC()

npoints = 15
npoints2 = 20

D1, D2 = grid(1e-5, 1, 1e-4, 10, npoints, npoints2)

E = model.E(D1, D2)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/50.)

model.fit(D1, D2, Efit)
model_no_gamma.fit(D1, D2, Efit)
#%timeit model.fit(D1, D2, Efit)
#%timeit model.fit(D1, D2, Efit, use_jacobian=False)
# With Jacobian
# noise /5.

# Without Jacobian (frequently has "covariance of parameters" warning)
# noise /5.
#%timeit model.fit(D1, D2, Efit, use_jacobian=False)

print(model)
print(model_no_gamma)

print(model.aic, model_no_gamma.aic)

fig = plt.figure(figsize=(8,3))

ax=fig.add_subplot(131)
plots.plot_colormap(D1, D2, E, ax=ax, title="True", cmap="viridis", vmin=0, vmax=1)

ax=fig.add_subplot(132)
plots.plot_colormap(D1, D2, Efit, ax=ax, title="Noisy", cmap="viridis", vmin=0, vmax=1)

ax=fig.add_subplot(133)
model.plot_colormap(D1, D2, ax=ax, title="Fit", vmin=0, vmax=1)

plt.tight_layout()
plt.show()


scatter_points = pd.DataFrame({'drug1.conc':D1, 'drug2.conc':D2, 'effect':Efit})

DD1, DD2 = grid(1e-8, 1e2, 1e-6, 1e3, npoints*2, npoints2*2)
truemodel.plot_surface_plotly(DD1, DD2, fname="musyc_gamma.html", scatter_points=scatter_points)
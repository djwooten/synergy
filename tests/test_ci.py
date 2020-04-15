import numpy as np
from synergy.single.hill import Hill
from synergy.combination.musyc import MuSyC
from synergy.combination.combination_index import CombinationIndex
from synergy.utils.dose_tools import grid

from matplotlib import pyplot as plt

E0, E1, E2, E3 = 1, 0, 0, 0
h1, h2 = 1., 1.
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 0., 0.

# Generate synthetic data with MuSyC (because h=1 and alpha=0, this model should look additive by CI)
model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

# Generate synthetic data with MuSyC (because alpha=1, this model should look synergistic by CI)
model2 = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=1., alpha21=1)

# Generate synthetic data with MuSyC (because Emax>0, CI will have a poor model fit for the single drugs)
model3 = MuSyC(E0=E0, E1=0.4, E2=0.4, E3=0.4, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

npoints = 8

D1, D2 = grid(1e-3, 1, 1e-2, 10, npoints, npoints, include_zero=True)

E = model.E(D1, D2)
E_2 = model2.E(D1, D2)
E_3 = model3.E(D1, D2)

ci = CombinationIndex()
ci2 = CombinationIndex()
ci3 = CombinationIndex()

ci.fit(D1, D2, E)
ci2.fit(D1, D2, E_2)
ci3.fit(D1, D2, E_3)


# Make plots
fig = plt.figure(figsize=(8,9))

# ---------------
# Model 1
# ---------------
ax = fig.add_subplot(3,2,1)
model.plot_colormap(D1, D2, ax=ax, title="Dose Surface", vmin=E3, vmax=E0)
ax.set_xticks([])

ax = fig.add_subplot(3,2,2)
ci.plot_colormap(ax=ax, title="CI", vmin=-0.01, vmax=0.01, neglog=True)
ax.set_xticks([])
ax.set_yticks([])

# ---------------
# Model 2
# ---------------

ax = fig.add_subplot(3,2,3)
model2.plot_colormap(D1, D2, ax=ax, title="Dose Surface", vmin=E3, vmax=E0)
ax.set_xticks([])

ax = fig.add_subplot(3,2,4)
ci2.plot_colormap(ax=ax, title="CI", center_on_zero=True, neglog=True)
ax.set_xticks([])
ax.set_yticks([])

# ---------------
# Model 3
# ---------------

ax = fig.add_subplot(3,2,5)
model3.plot_colormap(D1, D2, ax=ax, title="Dose Surface", vmin=E3, vmax=E0)

ax = fig.add_subplot(3,2,6)
ci3.plot_colormap(ax=ax, title="CI", center_on_zero=True, neglog=True)
ax.set_yticks([])

plt.tight_layout()
plt.show()
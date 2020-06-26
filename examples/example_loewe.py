import numpy as np
from synergy.single import Hill
from synergy.combination import MuSyC
from synergy.combination import Loewe
from synergy.utils.dose_tools import grid

from matplotlib import pyplot as plt

E0, E1, E2, E3 = 1, 0.5, 0.2, 0.1
h1, h2 = 1., 1.
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 0., 0.

gamma12, gamma21 = 1., 1.

model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)

model2 = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3*1.5, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=1, alpha21=alpha21*2, gamma12=gamma12, gamma21=gamma21)

drug1 = Hill(E0=E0, Emax=E1, h=h1, C=C1)
drug2 = Hill(E0=E0, Emax=E2, h=h2, C=C2)

npoints = 8

D1, D2 = grid(1e-3, 1, 1e-2, 10, npoints, npoints)

E = model.E(D1, D2)
E_2 = model2.E(D1, D2)

loewe = Loewe()
loewe2 = Loewe()

loewe.fit(D1, D2, E, drug1_model=drug1, drug2_model=drug2)
loewe2.fit(D1, D2, E_2, drug1_model=drug1, drug2_model=drug2)

fig = plt.figure(figsize=(7,6))

ax = fig.add_subplot(2,2,1)
model.plot_heatmap(D1, D2, ax=ax, title="Dose Surface", vmin=E3, vmax=E0)
ax.set_xticks([])

ax = fig.add_subplot(2,2,2)
loewe.plot_heatmap(ax=ax, title="Loewe", vmin=-0.01, vmax=0.01, neglog=True)
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(2,2,3)
model2.plot_heatmap(D1, D2, ax=ax, title="Dose Surface", vmin=E3, vmax=E0)

ax = fig.add_subplot(2,2,4)
loewe2.plot_heatmap(ax=ax, title="Loewe", center_on_zero=True, neglog=True)
ax.set_yticks([])

plt.tight_layout()
plt.show()
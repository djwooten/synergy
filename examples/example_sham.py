import numpy as np
from synergy.utils import sham
from synergy.single import Hill
from synergy.combination import Loewe
from synergy.combination import Bliss
from synergy.combination import Schindler
from synergy.combination import MuSyC

from matplotlib import pyplot as plt

E0 = 1
Emax = 0
h = 1.
C = 1e-1
drug = Hill(E0=E0, Emax=Emax, h=h, C=C)


npoints=8
dmax = 0.02
d = np.linspace(dmax/npoints,dmax,num=npoints)
d1, d2, E = sham(d, drug)

loewe = Loewe()
synergy = loewe.fit(d1, d2, E)

schindler = Schindler()
s_synergy = schindler.fit(d1, d2, E, drug1_model=loewe.drug1_model, drug2_model=loewe.drug2_model)

bliss = Bliss()
bsynergy = bliss.fit(d1, d2, E, drug1_model=loewe.drug1_model, drug2_model=loewe.drug2_model)

musyc = MuSyC()
musyc.fit(d1, d2, E)

fig = plt.figure(figsize=(12,3))

ax = fig.add_subplot(1,4,1)
musyc.plot_heatmap(d1, d2, ax=ax, logscale=False, title="Sham Surface")

ax = fig.add_subplot(1,4,2)
loewe.plot_heatmap(ax=ax, title="-log(Loewe)", vmin=-0.01, vmax=0.01, logscale=False, neglog=True)
ax.set_yticks([])

ax = fig.add_subplot(1,4,3)
schindler.plot_heatmap(ax=ax, title="Schindler", vmin=-0.01, vmax=0.01, logscale=False)
ax.set_yticks([])

ax = fig.add_subplot(1,4,4)
bliss.plot_heatmap(ax=ax, title="Bliss", center_on_zero=True, logscale=False)
ax.set_yticks([])

plt.tight_layout()
plt.show()
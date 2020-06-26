import numpy as np
from matplotlib import pyplot as plt


from synergy.utils import sham
from synergy.single import Hill
from synergy.combination import HSA
from synergy.utils import plots


hsa = HSA()
d1 = np.asarray([0,1,1], dtype=np.float64)
d2 = np.asarray([1,0,1], dtype=np.float64)
E = np.asarray([0.5, 0.4, 0.1])
synergy = hsa.fit(d1, d2, E)
print(synergy)


E0 = 1
Emax = 0
h = 1.
C = 1e-1
drug = Hill(E0=E0, Emax=Emax, h=h, C=C)

npoints=8
d = np.logspace(-3,0,num=npoints)
d1, d2, E = sham(d, drug)


synergy = hsa.fit(d1, d2, E)

fig = plt.figure(figsize=(6,3))

ax = fig.add_subplot(121)
plots.plot_heatmap(d1, d2, E, ax=ax, title="Data", cmap="viridis")

ax = fig.add_subplot(122)
hsa.plot_heatmap(ax=ax, title="HSA", center_on_zero=True)

plt.tight_layout()
plt.show()

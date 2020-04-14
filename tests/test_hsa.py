import numpy as np
from matplotlib import pyplot as plt


from synergy.utils.utils import sham
from synergy.single.hill import Hill
from synergy.combination.hsa import HSA
import synergy.utils.plots as plots


hsa = HSA()
d1 = np.asarray([0,1,1])
d2 = np.asarray([1,0,1])
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
plots.plot_colormap(d1, d2, E, ax=ax, title="Data")

ax = fig.add_subplot(122)
plots.plot_colormap(d1, d2, synergy, ax=ax, title="HSA")

plt.tight_layout()
plt.show()

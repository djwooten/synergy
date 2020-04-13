import numpy as np
from synergy.utils.utils import sham
from synergy.single.hill import Hill
from synergy.combination.loewe import Loewe
from synergy.combination.bliss import Bliss
from synergy.combination.schindler import Schindler
from synergy.combination.musyc import MuSyC

from matplotlib import pyplot as plt

E0 = 1
Emax = 0
h = 1.
C = 1e-1
drug = Hill(E0=E0, Emax=Emax, h=h, C=C)


npoints=8
dmax = 0.2
d = np.linspace(dmax/npoints,dmax,num=npoints)
d1, d2, E = sham(d, drug)

loewe = Loewe()
synergy = loewe.fit(d1, d2, E)

schindler = Schindler()
s_synergy = schindler.fit(d1, d2, E, drug1_model=loewe._drug1_model, drug2_model=loewe._drug2_model)

bliss = Bliss()
bsynergy = bliss.fit(d1, d2, E, drug1_model=loewe._drug1_model, drug2_model=loewe._drug2_model)

musyc = MuSyC()
musyc.fit(d1, d2, E)

fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot(1,3,1)
ax.pcolormesh(d1.reshape(npoints+1,npoints+1),d2.reshape(npoints+1,npoints+1),E.reshape(npoints+1,npoints+1))
ax.set_aspect('equal')
ax.set_title("Sham Surface")

ax = fig.add_subplot(1,3,2)
ax.pcolormesh(d1.reshape(npoints+1,npoints+1),d2.reshape(npoints+1,npoints+1),synergy.reshape(npoints+1,npoints+1), vmin=0.99, vmax=1.01)
ax.set_aspect('equal')
ax.set_yticks([])
ax.set_title("Loewe Synergy")

ax = fig.add_subplot(1,3,3)
ax.pcolormesh(d1.reshape(npoints+1,npoints+1),d2.reshape(npoints+1,npoints+1),s_synergy.reshape(npoints+1,npoints+1), vmin=-0.01, vmax=0.01)
ax.set_aspect('equal')
ax.set_yticks([])
ax.set_title("Schindler Synergy")

plt.tight_layout()
plt.show()
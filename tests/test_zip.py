import numpy as np
from matplotlib import pyplot as plt

from synergy.combination.zero_interaction_potency import ZIP
from synergy.combination.musyc import MuSyC
from synergy.single.hill import Hill

import synergy.utils.plots as plots


E0, E1, E2, E3 = 1., 0., 0., 0.
h1, h2 = 1., 1.
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 10., 1.

musyc = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

npoints = 20
d1 = np.logspace(-4,1,num=npoints)
d2 = np.logspace(-4,1,num=npoints)
D1, D2 = np.meshgrid(d1,d2)
D1 = D1.flatten()
D2 = D2.flatten()

E = musyc.E(D1, D2)


# Build ZIP model
model = ZIP()

#Efit = E*(1+(np.random.rand(len(D1))-0.5)/10.)
Efit = E

synergy = model.fit(D1, D2, Efit, drug1_model=Hill(E0,E1,h1,C1), drug2_model=Hill(E0,E2,h2,C2))
#synergy = model.fit(D1, D2, Efit)
print(model.drug1_model, model.drug2_model)

fig = plt.figure(figsize=(7,3))

ax=fig.add_subplot(121)
musyc.plot_colormap(D1, D2, ax=ax, title="Data")

ax=fig.add_subplot(122)
plots.plot_colormap(D1, D2, 100*synergy, ax=ax, title="ZIP")

plt.tight_layout()
plt.show()
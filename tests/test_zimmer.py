from matplotlib import pyplot as plt
import numpy as np
from synergy.combination.zimmer import Zimmer
import synergy.utils.plots as plots


h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
a12, a21 = -1., 1.
npoints = 10

model = Zimmer(h1=h1, h2=h2, C1=C1, C2=C2, a12=a12, a21=a21)

d1 = np.logspace(-3,0,num=npoints)
d2 = np.logspace(-2,0,num=npoints)
D1, D2 = np.meshgrid(d1,d2)
D1 = D1.flatten()
D2 = D2.flatten()

E = model._model(D1, D2, h1, h2, C1, C2, a12, a21)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/3.)

model.fit(D1, D2, Efit)
print(model)

fig = plt.figure(figsize=(10,3))

ax=fig.add_subplot(131)
plots.plot_colormap(D1, D2, E, ax=ax, title="True")

ax=fig.add_subplot(132)
plots.plot_colormap(D1, D2, Efit, ax=ax, title="Noisy")

ax=fig.add_subplot(133)
model.plot_colormap(D1, D2, ax=ax, title="Fit")

plt.tight_layout()
plt.show()
from matplotlib import pyplot as plt
import numpy as np
from synergy.combination.zimmer import Zimmer


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
Efit = E*(1+(np.random.rand(len(D1))-0.5)/1.)

model.fit(D1, D2, Efit)
print(model)

fig = plt.figure(figsize=(8,3))

ax=fig.add_subplot(131)
ax.set_title("True")
ax.pcolor(E.reshape((npoints,npoints)))
ax.set_aspect('equal')

ax=fig.add_subplot(132)
ax.set_title("Noisy")
ax.pcolor(Efit.reshape((npoints,npoints)))
ax.set_aspect('equal')

ax=fig.add_subplot(133)
ax.set_title("Fit")
ax.pcolor(model.E(D1, D2).reshape((npoints,npoints)))
ax.set_aspect('equal')

plt.show()
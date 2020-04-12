from matplotlib import pyplot as plt
import numpy as np
from synergy.combination.models import Zimmer


h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
a12, a21 = -1., 1.

model = Zimmer(h1=h1, h2=h2, C1=C1, C2=C2, a12=a12, a21=a21)

d1 = np.logspace(-3,0,num=20)
d2 = np.logspace(-2,0,num=20)
D1, D2 = np.meshgrid(d1,d2)
D1 = D1.flatten()
D2 = D2.flatten()

E = model._model(D1, D2, h1, h2, C1, C2, a12, a21)*(1+(np.random.rand(len(D1))-0.5)/2)

model.fit(D1, D2, E)
print(model)

fig = plt.figure()
ax=fig.add_subplot(121)
ax.pcolor(E.reshape((20,20)))

ax=fig.add_subplot(122)
ax.pcolor(model.E(D1, D2).reshape((20,20)))

plt.show()
import numpy as np
from synergy.combination import BRAID
from synergy.utils.dose_tools import grid
import pandas as pd
from matplotlib import pyplot as plt

E0, E1, E2, E3 = 0, 1, 1, 1
h1, h2 = 3, 3
C1, C2 = 1.5, 1.5
kappa = 0
delta = 1

xE0 = [0,0,0,0,0,0,0,0,0]
xE1 = [1,1,1,1,1,1,1,1,1]
xE2 = [1,1,1,1,1,0.55,1,1,0.75]
xE3 = [1,1,1,1,1,1,1,1,1]

xh1 = [3,1,1,3,3,3,3,3,2.7]
xh2 = [3,1,3,3,3,3,3,3,3.2]
xC1 = list(np.asarray([3,3,3,3,3,3,3,3,3])/2)
xC2 = list(np.asarray([3,3,3,3,3,3,3,3,3])/2-0.5)

#xkappa = [0,0,0,0.75,2.5,0,-0.3,-1.2,-0.5]
#xdelta = [1,1,1,1,1,1,1,1,1]

xkappa = [0,0,0,0.75,2.5,0,-0.3,-1.2,-0.5]
xdelta = [2,1,1,1,1,1,1,1,1]

npoints = 80

D1, D2 = grid(0,3,0,3,npoints,npoints,logscale=False)

fig = plt.figure(figsize=(8,8))

for i in range(9):
    ax = fig.add_subplot(3,3,1+i)

    E0 = xE0[i]
    E1 = xE1[i]
    E2 = xE2[i]
    E3 = xE3[i]

    h1 = xh1[i]
    h2 = xh2[i]
    C1 = xC1[i]
    C2 = xC2[i]

    kappa = xkappa[i]
    delta = xdelta[i]

    
    truemodel = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=kappa, delta=delta, variant="both")

    break

    truemodel.plot_colormap(D1, D2, logscale=False, cmap="RdYlGn_r", ax=ax, vmin=0, vmax=1)

D1, D2 = grid(C1/100,C1*100,C2/100,C2*100,8,8)

E = truemodel.E(D1, D2)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/200.)

model = BRAID(variant="delta")
model.fit(D1, D2, Efit, bootstrap_iterations=100)
print(model.summary())

#plt.tight_layout()
#plt.show()
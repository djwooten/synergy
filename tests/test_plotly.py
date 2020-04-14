from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import synergy.combination.musyc as musyc
import synergy.utils.plots as plots


E0, E1, E2, E3 = 0.5, 0.2, 0.1, -0.1
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 10.2, 1.1

model = musyc.MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

r1 = model.r1
r2 = model.r2

npoints = 8
npoints2 = 12

d1 = np.logspace(-3,0,num=npoints)/3.
d2 = np.logspace(-2,1,num=npoints2)
D1, D2 = np.meshgrid(d1,d2)
D1 = D1.flatten()
D2 = D2.flatten()


E = model._model(D1, D2, E0, E1, E2, E3, h1, h2, C1, C2, r1, r2, alpha12, alpha21)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/2.)

scatter_points = pd.DataFrame({'drug1.conc':D1, 'drug2.conc':D2, 'effect':Efit})
model.fit(D1, D2, Efit)


d1 = np.logspace(-3,0,num=10*npoints)/3.
d2 = np.logspace(-2,1,num=10*npoints2)
D1, D2 = np.meshgrid(d1,d2)

plots.plot_surface_plotly(D1, D2, model.E(D1,D2), scatter_points=scatter_points, xlabel="Drug1", ylabel="Drug2", zlabel="Effect", fname="plotly_musyc.html")
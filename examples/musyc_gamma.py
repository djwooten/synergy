from matplotlib import pyplot as plt
import numpy as np
from synergy.combination import MuSyC, MuSyCG
from synergy.utils import plots
from synergy.utils.dose_tools import grid
import pandas as pd

E0, E1, E2, E3 = 1, 0.5, 0.3, 0.
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
oalpha12, oalpha21 = 1., 1.
gamma12, gamma21 = 0.3, 0.3

alpha12 = MuSyCG._prime_to_alpha(oalpha12, C2, gamma12)
alpha21 = MuSyCG._prime_to_alpha(oalpha21, C1, gamma21)

model = MuSyCG(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, oalpha12=oalpha12, oalpha21=oalpha21, gamma12=gamma12, gamma21=gamma21)

truemodel = MuSyCG(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, oalpha12=oalpha12, oalpha21=oalpha21, gamma12=gamma12, gamma21=gamma21)

model_no_gamma = MuSyC()

npoints = 8
npoints2 = 12

D1, D2 = grid(C1/1e3, C1*1e3, C2/1e3, C2*1e3, npoints, npoints2)

E = model.E(D1, D2)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/5.)

if True:
    
    model.fit(D1, D2, Efit)
    model_no_gamma.fit(D1, D2, Efit)
    #%timeit model.fit(D1, D2, Efit)
    #%timeit model.fit(D1, D2, Efit, use_jacobian=False)
    # With Jacobian
    # noise /5.

    # Without Jacobian (frequently has "covariance of parameters" warning)
    # noise /5.
    #%timeit model.fit(D1, D2, Efit, use_jacobian=False)

    print(model)
    print(model_no_gamma)

    print(model.aic, model_no_gamma.aic)

    fig = plt.figure(figsize=(8,3))

    ax=fig.add_subplot(131)
    plots.plot_colormap(D1, D2, E, ax=ax, title="True", cmap="viridis", vmin=0, vmax=1)

    ax=fig.add_subplot(132)
    plots.plot_colormap(D1, D2, Efit, ax=ax, title="Noisy", cmap="viridis", vmin=0, vmax=1)

    ax=fig.add_subplot(133)
    model.plot_colormap(D1, D2, ax=ax, title="Fit", vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()


scatter_points = pd.DataFrame({'drug1.conc':D1, 'drug2.conc':D2, 'effect':Efit})

DD1, DD2 = grid(C1/1e5, C1*1e5, C2/1e5, C2*1e5, npoints*2, npoints2*2)
truemodel.plot_surface_plotly(DD1, DD2, fname="musyc_gamma.html", scatter_points=scatter_points)


model.plot_surface_plotly(DD1, DD2, fname="musyc_gamma_fit.html", scatter_points=scatter_points)

model_no_gamma.plot_surface_plotly(DD1, DD2, fname="musyc_nogamma_fit.html", scatter_points=scatter_points)
from matplotlib import pyplot as plt
import numpy as np

from synergy.combination import MuSyC
from synergy.utils import plots
from synergy.utils.dose_tools import make_dose_grid


# Shows that alpha and gamma are orthogonal synergy parameters

E0, E1, E2, E3 = 1, 0.3, 0.1, 0.0
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
# alpha12, alpha21 = 3.2, 1.1
# gamma12, gamma21 = 1, 1

npoints = 50
npoints2 = 50
d1, d2 = make_dose_grid(1e-3 / 3, 1 / 3, 1e-3, 10, npoints, npoints2, include_zero=False)

MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha12=1,
    alpha21=1,
    gamma12=1,
    gamma21=1,
).plot_surface_plotly(d1, d2, fname="1_1_1_1.html", auto_open=True)

MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha12=10,
    alpha21=0.1,
    gamma12=1,
    gamma21=1,
).plot_surface_plotly(d1, d2, fname="10_01_1_1.html", auto_open=True)

MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha12=1,
    alpha21=1,
    gamma12=10,
    gamma21=0.1,
).plot_surface_plotly(d1, d2, fname="1_1_10_01.html", auto_open=True)

MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha12=0.1,
    alpha21=10,
    gamma12=10,
    gamma21=0.1,
).plot_surface_plotly(d1, d2, fname="01_10_10_01.html", auto_open=True)

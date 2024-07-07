import numpy as np

from synergy.combination import MuSyC, MuSyC2
from synergy.combination.jacobians import musyc_jacobian as MJ
from synergy.combination.jacobians import musyc_jacobian_updated as MJ_U
from synergy.utils.dose_tools import make_dose_grid

E0 = 1
E1 = 0.5
E2 = 0.3
E3 = 0
h1 = 0.8
h2 = 2
C1 = 0.1
C2 = 0.1
alpha21 = 0.2
alpha12 = 10
gamma21 = 3
gamma12 = 0.7
r1r = 1
r2r = 1

d1, d2 = make_dose_grid(1e-3, 1e2, 1e-3, 1e2, 20, 20)


model1 = MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha21=alpha21,
    alpha12=alpha12,
    gamma21=gamma21,
    gamma12=gamma12,
)

model2 = MuSyC2(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha21=alpha21,
    alpha12=alpha12,
    gamma21=gamma21,
    gamma12=gamma12,
)

delta = model1.E(d1, d2) - model2.E(d1, d2)


j1 = MJ.jacobian(
    d1,
    d2,
    E0,
    E1,
    E2,
    E3,
    np.log(h1),
    np.log(h2),
    np.log(C1),
    np.log(C2),
    r1r,
    r2r,
    np.log(alpha12),
    np.log(alpha21),
    np.log(gamma12),
    np.log(gamma21),
)


j2 = MJ_U.jacobian(
    d1,
    d2,
    E0,
    E1,
    E2,
    E3,
    np.log(h1),
    np.log(h2),
    np.log(C1),
    np.log(C2),
    r1r,
    r2r,
    np.log(alpha12),
    np.log(alpha21),
    np.log(gamma12),
    np.log(gamma21),
)

delta_j = j1 - j2

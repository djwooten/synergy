import numpy as np
from matplotlib import pyplot as plt

from synergy.combination import MuSyC
from synergy.utils import plots
from synergy.utils.dose_tools import make_dose_grid

E0, E1, E2, E3 = 1, 0.2, 0.1, 0.4
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 3.2, 1.1
gamma12, gamma21 = 1, 1

model = MuSyC(
    E0=E0,
    E1=E1,
    E2=E2,
    E3=E3,
    h1=h1,
    h2=h2,
    C1=C1,
    C2=C2,
    alpha12=alpha12,
    alpha21=alpha21,
    gamma12=gamma12,
    gamma21=gamma21,
)

npoints = 8
npoints2 = 12

D1, D2 = make_dose_grid(1e-3 / 3, 1 / 3, 1e-2, 10, npoints, npoints2, include_zero=True)

E = model.E(D1, D2)
Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 5.0)

model.fit(D1, D2, Efit)
# %timeit model.fit(D1, D2, Efit)
# %timeit model.fit(D1, D2, Efit, use_jacobian=False)
# With Jacobian
# noise /5.
# 73.5 ms ± 965 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 63.7 ms ± 203 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Without Jacobian (frequently has "covariance of parameters" warning)
# noise /5.
# 26.1 ms ± 385 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 21.5 ms ± 422 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 29.6 ms ± 101 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 30.3 ms ± 284 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 20 ms ± 137 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 25.2 ms ± 343 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 25.8 ms ± 302 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# %timeit model.fit(D1, D2, Efit, use_jacobian=False)

print(model)

fig = plt.figure(figsize=(7, 7))

ax = fig.add_subplot(221)
plots.plot_heatmap(D1, D2, E, ax=ax, title="True", cmap="viridis")

ax = fig.add_subplot(222)
plots.plot_heatmap(D1, D2, Efit, ax=ax, title="Noisy", cmap="viridis")

ax = fig.add_subplot(223)
model.plot_heatmap(D1, D2, ax=ax, title="Fit")

ax = fig.add_subplot(224)
model.plot_residual_heatmap(D1, D2, Efit, ax=ax, title="Residuals", center_on_zero=True)

plt.tight_layout()
plt.show()

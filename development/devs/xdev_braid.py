import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from synergy.combination import BRAID
from synergy.utils.dose_tools import make_dose_grid

E0, E1, E2, E3 = 0, 1, 1, 1
h1, h2 = 3, 3
C1, C2 = 1.5, 1.5

model1 = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=0, delta=10)
model2 = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=10, delta=1)

D1, D2 = make_dose_grid(C1 / 100, C1 * 100, C2 / 100, C2 * 100, 8, 8)


fig = plt.figure(figsize=(8, 5))

ax = fig.add_subplot(121)
model1.plot_heatmap(D1, D2, ax=ax)

ax = fig.add_subplot(122)
model2.plot_heatmap(D1, D2, ax=ax)

plt.show()

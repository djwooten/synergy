import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from synergy.combination import ZIP
from synergy.combination import MuSyC
from synergy.single import Hill
from synergy.combination import BRAID

from synergy.utils import plots
from synergy.utils.dose_tools import make_dose_grid


E0, E1, E2, E3 = 1.0, 0.0, 0.0, 0.0
h1, h2 = 1.0, 1.0
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 10.0, 1.0
gamma12, gamma21 = 1, 1

musyc = MuSyC(
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

npoints1 = 8
npoints2 = 10
D1, D2 = make_dose_grid(1e-4, 10, 1e-4, 10, npoints1, npoints2, include_zero=True)

E = musyc.E(D1, D2)


# Build and fit ZIP model
zip_model = ZIP(synergyfinder=True)
zip_model.fit(D1, D2, E)

# Build and fit BRAID model
# braid_model = BRAID()
# braid_model.fit(D1, D2, E, bootstrap_iterations=100)


sfdf = pd.read_csv("synergyfinder_output.csv")
sfdf.sort_values(by=["d2", "d1"], inplace=True)

print(
    "Correlation between this package and synergyfinder = %f"
    % np.corrcoef(zip_model.synergy, sfdf["synergy"])[0, 1]
)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.scatter(zip_model.synergy * 100, sfdf["synergy"])
ax.set_xlabel("Python ZIP synergy (x100)")
ax.set_ylabel("R synergyfinder ZIP synergy")
ax.set_title("Comparison with synergyfinder")
plt.tight_layout()
plt.savefig("python_vs_synergyfinder2.pdf")
plt.close()
# plt.show()

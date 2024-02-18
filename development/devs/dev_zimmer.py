import numpy as np
from synergy.combination import Zimmer
from synergy.utils.dose_tools import make_dose_grid


h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
a12, a21 = -1.0, 1.0
npoints = 10

truemodel = Zimmer(h1=h1, h2=h2, C1=C1, C2=C2, a12=a12, a21=a21)

D1, D2 = make_dose_grid(1e-3, 1, 1e-2, 1, npoints, npoints, include_zero=True)
E = truemodel._model(D1, D2, h1, h2, C1, C2, a12, a21)


Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 5.0)

model = Zimmer()
model.fit(D1, D2, Efit, bootstrap_iterations=100)
print(model)
if model.converged:
    print(model.get_parameter_range().T)

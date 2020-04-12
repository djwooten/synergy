from matplotlib import pyplot as plt
import numpy as np
from synergy.single.hill import Hill

model = Hill()

E0 = 1
Emax = 0
h = 2.3
C = 1e-2

dose = np.logspace(-2,0,num=20)
E = model._model(dose, E0, Emax, h, C)*(1+(np.random.rand(len(dose))-0.5)/10)

model.fit(dose, E)
print(model.get_parameters())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dose,E)

dose = np.logspace(-2,0)
E = model.E(dose)
ax.plot(dose,E)
ax.set_xscale('log')

plt.show()
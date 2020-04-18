from matplotlib import pyplot as plt
import numpy as np
from synergy.single import Hill

E0 = 1
Emax = 0
h = 2.3
C = 1e-2

model = Hill(E0, Emax, h, C)

d = np.logspace(-3,0,num=10)
E = model.E(d)
#E_fit = E*(1+(np.random.rand(len(E))-0.5)/3)
E_fit = E + 0.1*(2*np.random.rand(len(E))-1)

print(model.fit(d, E_fit, bootstrap_iterations=100))

print("\n")

print(model)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(d,E_fit)

d = np.logspace(-3,0)
ax.plot(d,model.E(d))

ax.set_xscale('log')
plt.show()
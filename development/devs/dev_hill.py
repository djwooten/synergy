import numpy as np
from matplotlib import pyplot as plt

from synergy.single import Hill, Hill_2P, Hill_CI

E0 = 1
Emax = 0
h = 2.3
C = 1e-2

noise = 0.05

d = np.logspace(-3, 0, num=20)

truemodel = Hill(E0, Emax, h, C)
E = truemodel.E(d)

# E_fit = E*(1+(np.random.rand(len(E))-0.5)/3)
E_fit = E + noise * (E0 - Emax) * (2 * np.random.rand(len(E)) - 1)

model = Hill()
model.fit(d, E_fit, bootstrap_iterations=100)

print("\n")
print(model)
print(model.get_confidence_intervals())


if True:
    model_2P = Hill_2P(E0=E0, Emax=Emax)
    model_2P.fit(d, E_fit, bootstrap_iterations=100)

    print("\n")
    print(model_2P)
    print(model_2P.get_confidence_intervals())

    model_CI = Hill_CI()
    model_CI.fit(d, E_fit, bootstrap_iterations=100)

    print("\n")
    print(model_CI)
    print(model_CI.get_confidence_intervals())


if False:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d, E_fit)

    d = np.logspace(-3, 0)
    ax.plot(d, model.E(d), label="Hill()", alpha=0.5, lw=2)
    ax.plot(d, model_2P.E(d), label="Hill_2P()", alpha=0.5, lw=2)
    ax.plot(d, model_CI.E(d), label="Hill_CI()", alpha=0.5, lw=2)

    ax.plot(d, truemodel.E(d), lw=5, alpha=0.1, c="k", label="True")

    ax.legend()
    ax.set_xscale("log")
    plt.show()

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d, E_fit)

    d = np.logspace(-3, 0)
    for i in range(model.bootstrap_parameters.shape[0]):
        parms = model.bootstrap_parameters[i, :]
        ax.plot(d, model._model(d, *parms), alpha=0.1, c="k")

    ax.plot(d, model.E(d), lw=5, alpha=0.5, c="b", label="Best Fit")
    ax.plot(d, truemodel.E(d), lw=5, alpha=0.5, c="r", label="True")

    ax.set_xscale("log")
    ax.legend()
    plt.show()

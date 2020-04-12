from matplotlib import pyplot as plt
import numpy as np
import synergy.combination.musyc as musyc
import synergy.combination.musyc_jacobian as mj


E0, E1, E2, E3 = 1, 0.3, 0.1, 0.4
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 3.2, 0.1

model = musyc.MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

r1 = model.r1
r2 = model.r2

npoints = 20

d1 = np.logspace(-3,0,num=npoints)
d2 = np.logspace(-2,0,num=npoints)
D1, D2 = np.meshgrid(d1,d2)
D1 = D1.flatten()
D2 = D2.flatten()


E = model._model(D1, D2, E0, E1, E2, E3, h1, h2, C1, C2, r1, r2, alpha12, alpha21)
Efit = E*(1+(np.random.rand(len(D1))-0.5)/50.)

model.fit(D1, D2, Efit)
#%timeit model.fit(D1, D2, Efit)
# With Jacobian
# noise /5.
# 31.5 ms ± 213 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 52.5 ms ± 202 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 22.4 ms ± 366 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Without Jacobian
# noise /5.
# 20 ms ± 137 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 25.2 ms ± 343 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 25.8 ms ± 302 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
#%timeit model.fit(D1, D2, Efit, use_jacobian=False)


jac = mj.jacobian(D1, D2, E0, E1, E2, E3, np.log10(h1), np.log10(h2), np.log10(C1), np.log10(C2), 100., 100., np.log10(alpha12), np.log10(alpha21))



print(model)

fig = plt.figure(figsize=(8,3))

ax=fig.add_subplot(131)
ax.set_title("True")
ax.pcolor(E.reshape((npoints,npoints)))
ax.set_aspect('equal')

ax=fig.add_subplot(132)
ax.set_title("Noisy")
ax.pcolor(Efit.reshape((npoints,npoints)))
ax.set_aspect('equal')

ax=fig.add_subplot(133)
ax.set_title("Fit")
ax.pcolor(model.E(D1, D2).reshape((npoints,npoints)))
ax.set_aspect('equal')

plt.show()
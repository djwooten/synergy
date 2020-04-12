from sympy import *
import numpy as np


d, E0, Emax, h, C = symbols('d E0 Emax h C')

# Hill equation
f = E0 + (Emax-E0)*d**(10**h)/((10**C)**(10**h)+d**(10**h))
fE0 = diff(f,E0)
fEmax = diff(f,Emax)
fh = diff(f,h)
fC = diff(f,C)


# MuSyC
d1, d2 = symbols('d1 d2')
h1, h2 = symbols('h1 h2')
E1, E2, E3 = symbols('E1 E2 E3')
r1, r1r = symbols('r1 r1r')
r2, r2r = symbols('r2 r2r')
alpha12, alpha21 = symbols('alpha12 alpha21')
gamma12, gamma21 = symbols('gamma12 gamma21')

U = r1r*r2r*(r1*(alpha21*d1)**h1 + r1r + r2*(alpha12*d2)**h2 + r2r)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)

A1 = r1*r2r*(d1**h1*r1*(alpha21*d1)**h1 + d1**h1*r1r + d1**h1*r2r + d2**h2*r2*(alpha21*d1)**h1)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)

A2 = r1r*r2*(d1**h1*r1*(alpha12*d2)**h2 + d2**h2*r1r + d2**h2*r2*(alpha12*d2)**h2 + d2**h2*r2r)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)

A12 = 1-(U+A1+A2)
    
f = U*E0 + A1*E1 + A2*E2 + A12*E3

fr1 = diff(f,r1)
fr2 = diff(f,r2)
fr1r = diff(f,r1r)
fr2r = diff(f,r2r)
fh1 = diff(f,r1)
fh2 = diff(f,r2)
fE0 = U
fE1 = A1
fE2 = A2
fE3 = A12
falpha12 = diff(f, alpha12)
falpha21 = diff(f, alpha21)


def eval_falpha12(d1, d2, E0, E1, E2, E3, h1, h2, r1, r1r, r2, r2r, alpha12, alpha21):
    return E0*r1r*r2r*(r1*(alpha21*d1)**h1 + r1r + r2*(alpha12*d2)**h2 + r2r)*(-d1**h1*h2*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d1**h1*h2*r1*r1r*r2*(alpha12*d2)**h2/alpha12 - d1**h1*h2*r1*r2*r2r*(alpha12*d2)**h2/alpha12 - d2**h2*h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d2**h2*h2*r1r*r2**2*(alpha12*d2)**h2/alpha12 - h2*r1r*r2*r2r*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)**2 + E0*h2*r1r*r2*r2r*(alpha12*d2)**h2/(alpha12*(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)) + E1*r1*r2r*(d1**h1*r1*(alpha21*d1)**h1 + d1**h1*r1r + d1**h1*r2r + d2**h2*r2*(alpha21*d1)**h1)*(-d1**h1*h2*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d1**h1*h2*r1*r1r*r2*(alpha12*d2)**h2/alpha12 - d1**h1*h2*r1*r2*r2r*(alpha12*d2)**h2/alpha12 - d2**h2*h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d2**h2*h2*r1r*r2**2*(alpha12*d2)**h2/alpha12 - h2*r1r*r2*r2r*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)**2 + E2*r1r*r2*(d1**h1*h2*r1*(alpha12*d2)**h2/alpha12 + d2**h2*h2*r2*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2) + E2*r1r*r2*(d1**h1*r1*(alpha12*d2)**h2 + d2**h2*r1r + d2**h2*r2*(alpha12*d2)**h2 + d2**h2*r2r)*(-d1**h1*h2*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d1**h1*h2*r1*r1r*r2*(alpha12*d2)**h2/alpha12 - d1**h1*h2*r1*r2*r2r*(alpha12*d2)**h2/alpha12 - d2**h2*h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d2**h2*h2*r1r*r2**2*(alpha12*d2)**h2/alpha12 - h2*r1r*r2*r2r*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)**2 + E3*(-r1*r2r*(d1**h1*r1*(alpha21*d1)**h1 + d1**h1*r1r + d1**h1*r2r + d2**h2*r2*(alpha21*d1)**h1)*(-d1**h1*h2*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d1**h1*h2*r1*r1r*r2*(alpha12*d2)**h2/alpha12 - d1**h1*h2*r1*r2*r2r*(alpha12*d2)**h2/alpha12 - d2**h2*h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d2**h2*h2*r1r*r2**2*(alpha12*d2)**h2/alpha12 - h2*r1r*r2*r2r*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)**2 - r1r*r2*(d1**h1*h2*r1*(alpha12*d2)**h2/alpha12 + d2**h2*h2*r2*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2) - r1r*r2*(d1**h1*r1*(alpha12*d2)**h2 + d2**h2*r1r + d2**h2*r2*(alpha12*d2)**h2 + d2**h2*r2r)*(-d1**h1*h2*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d1**h1*h2*r1*r1r*r2*(alpha12*d2)**h2/alpha12 - d1**h1*h2*r1*r2*r2r*(alpha12*d2)**h2/alpha12 - d2**h2*h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d2**h2*h2*r1r*r2**2*(alpha12*d2)**h2/alpha12 - h2*r1r*r2*r2r*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)**2 - r1r*r2r*(r1*(alpha21*d1)**h1 + r1r + r2*(alpha12*d2)**h2 + r2r)*(-d1**h1*h2*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d1**h1*h2*r1*r1r*r2*(alpha12*d2)**h2/alpha12 - d1**h1*h2*r1*r2*r2r*(alpha12*d2)**h2/alpha12 - d2**h2*h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1/alpha12 - d2**h2*h2*r1r*r2**2*(alpha12*d2)**h2/alpha12 - h2*r1r*r2*r2r*(alpha12*d2)**h2/alpha12)/(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)**2 - h2*r1r*r2*r2r*(alpha12*d2)**h2/(alpha12*(d1**h1*r1**2*r2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d1**h1*r1**2*r2r*(alpha21*d1)**h1 + d1**h1*r1*r1r*r2*(alpha12*d2)**h2 + d1**h1*r1*r1r*r2r + d1**h1*r1*r2*r2r*(alpha12*d2)**h2 + d1**h1*r1*r2r**2 + d2**h2*r1*r1r*r2*(alpha21*d1)**h1 + d2**h2*r1*r2**2*(alpha12*d2)**h2*(alpha21*d1)**h1 + d2**h2*r1*r2*r2r*(alpha21*d1)**h1 + d2**h2*r1r**2*r2 + d2**h2*r1r*r2**2*(alpha12*d2)**h2 + d2**h2*r1r*r2*r2r + r1*r1r*r2r*(alpha21*d1)**h1 + r1r**2*r2r + r1r*r2*r2r*(alpha12*d2)**h2 + r1r*r2r**2)))


nE0, nE1, nE2, nE3 = 1, 0.4, 0.2, -0.1
nh1, nh2 = 0.8, 2.5
nr1, nr2 = 1., 1.
nC1, nC2 = 0.15, 0.08
nr1r = nr1*nC1**nh1
nr2r = nr2*nC2**nh2
nalpha12, nalpha21 = 0.01, 1.8

nd1 = np.logspace(-2,1,num=10)
nd2 = np.logspace(-2,1,num=10)
D1, D2 = np.meshgrid(nd1,nd2)
D1 = D1.flatten()
D2 = D2.flatten()

ff = lambdify((d1, d2, E0, E1, E2, E3, h1, h2, r1, r1r, r2, r2r, alpha12, alpha21), falpha12, "numpy")


#%timeit eval_falpha12(D1, D2, nE0, nE1, nE2, nE3, nh1, nh2, nr1, nr1r, nr2, nr2r, nalpha12, nalpha21)
#2.43 ms ± 15.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

#%timeit ff(D1, D2, nE0, nE1, nE2, nE3, nh1, nh2, nr1, nr1r, nr2, nr2r, nalpha12, nalpha21)
# 2.45 ms ± 21.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
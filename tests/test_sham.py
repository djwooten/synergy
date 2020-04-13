import numpy as np
from synergy.utils.utils import sham
from synergy.combination.loewe import Loewe
from synergy.combination.bliss import Bliss

E0 = 1
Emax = 0
h = 2.3
C = 1e-1

npoints=8

#d = np.logspace(-2,0,num=npoints)
d = np.linspace(10e-2,1,num=npoints)
d1, d2, E = sham(d, E0, Emax, h, C)

loewe = Loewe()
synergy = loewe.fit(d1, d2, E)

bliss = Bliss()
bsynergy = bliss.fit(d1, d2, E, drug1_model=loewe._drug1_model, drug2_model=loewe._drug2_model)


D1, D2, bnull = bliss.null_E(d, d)
b_null_synergy = bliss.fit(D1, D2, bnull, drug1_model=loewe._drug1_model, drug2_model=loewe._drug2_model)
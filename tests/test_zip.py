def test_zip():
    pass
import numpy as np
from synergy.combination import ZIP
from synergy.utils.dose_tools import grid
from synergy.single import Hill_2P
from synergy.utils import sham
h = 2.3
C = 1e-2

drug = Hill_2P(h=h, C=C)
d = np.logspace(-3,1,num=10)

D1, D2, E = sham(d, drug)

model2 = ZIP()

synergy = model2.fit(D1, D2, E)
assert np.max(synergy)>0.157




import numpy as np
from synergy.combination import ZIP
from synergy.utils.dose_tools import grid
from synergy.single import Hill_2P
from synergy.utils import sham
h = 2.3
C = 1e-2

drug = Hill_2P(E0=-2, Emax=10, h=h, C=C)
d = np.logspace(-3,1,num=10)

D1, D2, E = sham(d, drug)

model = ZIP()

Emax = -2
E0 = 10
#synergy = model.fit(D1, D2, (E-E0)/(Emax-E0))
synergy = model.fit(D1, D2, E)
#assert np.max(synergy)>0.157
def test_zip():
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

    model = ZIP()

    synergy = model.fit(D1, D2, E)
    assert np.max(synergy)>0.157

def test_musyc():
    import numpy as np
    from synergy.combination import MuSyC
    from synergy.utils.dose_tools import grid


    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    alpha12, alpha21 = 3.2, 1.1

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21)

    npoints = 8
    npoints2 = 12

    D1, D2 = grid(1e-3,1e0,1e-2,1e1,npoints,npoints2)

    E = model.E(D1, D2)
    Efit = E*(1+(np.random.rand(len(D1))-0.5)/10.)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9
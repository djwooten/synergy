def test_musyc():
    import numpy as np
    from synergy.combination import MuSyC
    from synergy.utils.dose_tools import grid


    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    alpha12, alpha21 = 3.2, 1.1
    gamma12, gamma21 = 4.1, 0.5

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)

    npoints = 8
    npoints2 = 12

    D1, D2 = grid(1e-3,1e0,1e-2,1e1,npoints,npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E*(1+(np.random.rand(len(D1))-0.5)/10.)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9

def test_musyc_no_gamma():
    import numpy as np
    from synergy.combination import MuSyC
    from synergy.utils.dose_tools import grid


    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    alpha12, alpha21 = 3.2, 1.1

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, variant="no_gamma")

    npoints = 8
    npoints2 = 12

    D1, D2 = grid(1e-3,1e0,1e-2,1e1,npoints,npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E*(1+(np.random.rand(len(D1))-0.5)/10.)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9

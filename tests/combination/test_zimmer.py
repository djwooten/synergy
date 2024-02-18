def test_zimmer():
    import numpy as np
    from synergy.combination import Zimmer
    from synergy.utils.dose_tools import make_dose_grid

    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    a12, a21 = -0.5, 1.2

    model = Zimmer(h1=h1, h2=h2, C1=C1, C2=C2, a12=a12, a21=a21)

    npoints = 8
    npoints2 = 12

    D1, D2 = make_dose_grid(1e-3, 1e0, 1e-2, 1e1, npoints, npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 10.0)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9

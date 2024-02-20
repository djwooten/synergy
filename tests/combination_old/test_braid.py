def test_braid_kappa():
    import numpy as np
    from synergy.combination import BRAID
    from synergy.utils.dose_tools import make_dose_grid

    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    kappa = 1
    delta = 0.5

    model = BRAID(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=1, variant="kappa")

    npoints = 8
    npoints2 = 12

    D1, D2 = make_dose_grid(1e-3, 1e0, 1e-2, 1e1, npoints, npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 10.0)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9


def test_braid_delta():
    import numpy as np
    from synergy.combination import BRAID
    from synergy.utils.dose_tools import make_dose_grid

    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    delta = 0.5

    model = BRAID(
        E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, delta=delta, variant="delta"
    )

    npoints = 8
    npoints2 = 12

    D1, D2 = make_dose_grid(1e-3, 1e0, 1e-2, 1e1, npoints, npoints2)

    np.random.seed(1)
    E = model.E(D1, D2)
    Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 10.0)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9


def test_braid_both():
    import numpy as np
    from synergy.combination import BRAID
    from synergy.utils.dose_tools import make_dose_grid

    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    kappa = 1
    delta = 0.5

    model = BRAID(
        E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, kappa=1, delta=delta, variant="both"
    )

    npoints = 8
    npoints2 = 12

    D1, D2 = make_dose_grid(1e-3, 1e0, 1e-2, 1e1, npoints, npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 10.0)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9

def test_musyc():
    import numpy as np
    from synergy.combination import MuSyC
    from synergy.utils.dose_tools import make_dose_grid

    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    alpha12, alpha21 = 3.2, 1.1
    gamma12, gamma21 = 4.1, 0.5

    model = MuSyC(
        E0=E0,
        E1=E1,
        E2=E2,
        E3=E3,
        h1=h1,
        h2=h2,
        C1=C1,
        C2=C2,
        alpha12=alpha12,
        alpha21=alpha21,
        gamma12=gamma12,
        gamma21=gamma21,
    )

    npoints = 8
    npoints2 = 12

    D1, D2 = make_dose_grid(1e-3, 1e0, 1e-2, 1e1, npoints, npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 10.0)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9


def test_musyc_no_gamma():
    import numpy as np
    from synergy.combination import MuSyC
    from synergy.utils.dose_tools import make_dose_grid

    E0, E1, E2, E3 = 1, 0.2, 0.1, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 1e-2, 1e-1
    alpha12, alpha21 = 3.2, 1.1

    model = MuSyC(
        E0=E0,
        E1=E1,
        E2=E2,
        E3=E3,
        h1=h1,
        h2=h2,
        C1=C1,
        C2=C2,
        alpha12=alpha12,
        alpha21=alpha21,
        fit_gamma="no_gamma",
    )

    npoints = 8
    npoints2 = 12

    D1, D2 = make_dose_grid(1e-3, 1e0, 1e-2, 1e1, npoints, npoints2)

    np.random.seed(0)
    E = model.E(D1, D2)
    Efit = E * (1 + (np.random.rand(len(D1)) - 0.5) / 10.0)

    model.fit(D1, D2, Efit)

    assert model.r_squared > 0.9


def test_musyc_higher():
    import numpy as np
    from synergy.utils import dose_tools
    from synergy.higher import MuSyC
    from synergy.utils import plots

    E_params = [2, 1, 1, 1, 1, 0, 0, 0]
    h_params = [2, 1, 0.8]
    C_params = [0.1, 0.01, 0.1]
    alpha_params = [2, 3, 1, 1, 0.7, 0.5, 2, 1, 1]
    gamma_params = [0.4, 2, 1, 2, 0.7, 3, 2, 0.5, 2]

    params = E_params + h_params + C_params + alpha_params + gamma_params

    truemodel = MuSyC()
    truemodel.parameters = params

    d = dose_tools.make_dose_grid_multi((1e-3, 1e-3, 1e-3), (1, 1, 1), (6, 6, 6), include_zero=True)

    d1 = d[:, 0]
    d2 = d[:, 1]
    d3 = d[:, 2]
    np.random.seed(0)
    E = truemodel.E(d)
    noise = 0.05
    E_fit = E + noise * (E_params[0] - E_params[-1]) * (2 * np.random.rand(len(E)) - 1)

    model = MuSyC(E_bounds=(0, 2), h_bounds=(1e-3, 1e3), alpha_bounds=(1e-5, 1e5), gamma_bounds=(1e-5, 1e5))
    model.fit(d, E_fit)

    assert model.r_squared > 0.9

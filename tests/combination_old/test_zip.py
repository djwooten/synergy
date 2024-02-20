def test_zip():
    import numpy as np
    from synergy.combination import ZIP
    from synergy.utils.dose_tools import make_dose_grid
    from synergy.single import Hill_2P
    from synergy.utils import sham

    h = 2.3
    C = 1e-2

    drug = Hill_2P(h=h, C=C)
    d = np.logspace(-3, 1, num=10)

    D1, D2, E = sham(d, drug)

    model2 = ZIP()

    synergy = model2.fit(D1, D2, E)
    assert np.max(synergy) > 0.15

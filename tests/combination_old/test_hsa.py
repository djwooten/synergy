def test_HSA_sham():
    import numpy as np
    from synergy.combination import HSA
    from tests.testing_utils.synthetic_data import sham

    d1, d2, E = sham()

    model = HSA()

    synergy = model.fit(d1, d2, E)
    assert np.nanmean(synergy) > 0.08


def test_HSA_sham_3():
    import numpy as np
    from synergy.higher import HSA
    from tests.testing_utils.synthetic_data import sham_3

    d, E = sham_3()

    model = HSA()

    synergy = model.fit(d, E)
    assert np.nanmean(synergy) > 0.08

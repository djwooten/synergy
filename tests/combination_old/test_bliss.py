def test_bliss_sham():
    import numpy as np
    from synergy.combination import Bliss
    from tests.testing_utils.synthetic_data import sham

    d1, d2, E = sham()

    model = Bliss()

    synergy = model.fit(d1, d2, E)
    assert np.nanmean(synergy) < 0.1


def test_bliss_msp():
    import numpy as np
    from synergy.combination import Bliss
    from tests.testing_utils.synthetic_data import bliss_independent

    d1, d2, E = bliss_independent()

    model = Bliss()

    synergy = model.fit(d1, d2, E)
    assert np.nanmax(np.abs(synergy)) < 0.12


def test_bliss_sham_3():
    import numpy as np
    from synergy.higher import Bliss
    from tests.testing_utils.synthetic_data import sham_3

    d, E = sham_3()

    model = Bliss()

    synergy = model.fit(d, E)
    assert np.nanmean(synergy) < 0.1


def test_bliss_msp_3():
    import numpy as np
    from synergy.higher import Bliss
    from tests.testing_utils.synthetic_data import bliss_independent_3

    d, E = bliss_independent_3(noise=0.03)

    model = Bliss()

    synergy = model.fit(d, E)
    assert np.nanmax(np.abs(synergy)) < 0.11

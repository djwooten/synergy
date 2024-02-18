def test_CI_sham():
    import numpy as np
    from synergy.combination import CombinationIndex
    from tests.testing_utils.synthetic_data import sham

    d1, d2, E = sham()

    model = CombinationIndex()

    synergy = model.fit(d1, d2, E)
    assert np.abs(np.nanmean(np.log(synergy))) < 0.1


def test_CI_msp():
    import numpy as np
    from synergy.combination import CombinationIndex
    from tests.testing_utils.synthetic_data import bliss_independent

    d1, d2, E = bliss_independent()

    model = CombinationIndex()

    synergy = model.fit(d1, d2, E)
    assert np.nanmax(np.abs(np.log(synergy))) > 1


def test_CI_sham_3():
    import numpy as np
    from synergy.higher import CombinationIndex
    from tests.testing_utils.synthetic_data import sham_3

    d, E = sham_3()

    model = CombinationIndex()

    synergy = model.fit(d, E)
    assert np.abs(np.nanmean(np.log(synergy))) < 0.1


def test_CI_msp_3():
    import numpy as np
    from synergy.higher import CombinationIndex
    from tests.testing_utils.synthetic_data import bliss_independent_3

    d, E = bliss_independent_3()

    model = CombinationIndex()

    synergy = model.fit(d, E)
    assert np.nanmax(np.abs(np.log(synergy))) > 1

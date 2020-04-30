def test_loewe_sham():
    import numpy as np
    from synergy.combination import Loewe
    from synergy.datasets import sham

    d1, d2, E = sham()

    model = Loewe()

    synergy = model.fit(d1, d2, E)
    assert np.abs(np.nanmean(np.log(synergy)))<0.1

def test_loewe_msp():
    import numpy as np
    from synergy.combination import Loewe
    from synergy.datasets import bliss_independent

    d1, d2, E = bliss_independent()

    model = Loewe()

    synergy = model.fit(d1, d2, E)
    assert np.nanmax(np.abs(np.log(synergy)))>1





def test_loewe_sham_3():
    import numpy as np
    from synergy.higher import Loewe
    from synergy.datasets import sham_3

    d, E = sham_3()

    model = Loewe()

    synergy = model.fit(d, E)
    assert np.abs(np.nanmean(np.log(synergy)))<0.1

def test_loewe_msp_3():
    import numpy as np
    from synergy.higher import Loewe
    from synergy.datasets import bliss_independent_3

    d, E = bliss_independent_3()

    model = Loewe()

    synergy = model.fit(d, E)
    assert np.nanmax(np.abs(np.log(synergy)))>1


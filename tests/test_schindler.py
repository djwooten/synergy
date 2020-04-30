def test_schindler_sham():
    import numpy as np
    from synergy.combination import Schindler
    from synergy.datasets import linear_isobole

    d1, d2, E = linear_isobole()

    model = Schindler()

    synergy = model.fit(d1, d2, E)
    assert np.abs(np.nanmean(synergy))<0.1

def test_schindler_msp():
    import numpy as np
    from synergy.combination import Schindler
    from synergy.datasets import bliss_independent

    d1, d2, E = bliss_independent()

    model = Schindler()

    synergy = model.fit(d1, d2, E)
    assert np.nanmax(np.abs(synergy))>0.1






def test_schindler_sham_3():
    import numpy as np
    from synergy.higher import Schindler
    from synergy.datasets import linear_isobole_3

    d, E = linear_isobole_3()

    model = Schindler()

    synergy = model.fit(d, E)
    assert np.abs(np.nanmean(synergy))<0.1

def test_schindler_msp_3():
    import numpy as np
    from synergy.higher import Schindler
    from synergy.datasets import bliss_independent_3

    d, E = bliss_independent_3()

    model = Schindler()

    synergy = model.fit(d, E)
    assert np.nanmax(np.abs(synergy))>0.1


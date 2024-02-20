def test_heatmap_replicates():
    import numpy as np
    from tests.testing_utils.synthetic_data import sham
    from synergy.utils.plots import plot_heatmap

    d1, d2, E = sham()
    d1 = np.hstack([d1, d1])
    d2 = np.hstack([d2, d2])
    E = np.hstack([E, E * 2])

    plot_heatmap(d1, d2, E, fname="x.pdf")
    assert 1 == 1


def test_heatmap_linearscale():
    import numpy as np
    from tests.testing_utils.synthetic_data import sham
    from synergy.utils.plots import plot_heatmap

    d1, d2, E = sham()
    d1 = np.hstack([d1, d1])
    d2 = np.hstack([d2, d2])
    E = np.hstack([E, E * 2])

    plot_heatmap(d1, d2, E, fname="x.pdf", logscale=False)
    assert 1 == 1


def test_plotly_replicates():
    import numpy as np
    from tests.testing_utils.synthetic_data import sham
    from synergy.utils.plots import plot_surface_plotly

    d1, d2, E = sham()
    d1 = np.hstack([d1, d1])
    d2 = np.hstack([d2, d2])
    E = np.hstack([E, E * 2])

    plot_surface_plotly(d1, d2, E, fname="x.html")
    assert 1 == 1


def test_plotly_linearscale():
    import numpy as np
    from tests.testing_utils.synthetic_data import sham
    from synergy.utils.plots import plot_surface_plotly

    d1, d2, E = sham()
    d1 = np.hstack([d1, d1])
    d2 = np.hstack([d2, d2])
    E = np.hstack([E, E * 2])

    plot_surface_plotly(d1, d2, E, fname="x.html", logscale=False)
    assert 1 == 1


def test_parametric_model_plotly_replicates():
    from synergy.utils.dose_tools import make_dose_grid
    from synergy.combination import MuSyC

    d1, d2 = make_dose_grid(0.01, 10, 0.01, 10, 5, 5, include_zero=True, replicates=3)
    model = MuSyC(
        E0=1,
        E1=0.2,
        E2=0.5,
        E3=0.1,
        h1=1.4,
        h2=0.9,
        C1=0.2,
        C2=0.2,
        alpha12=2,
        alpha21=0.5,
        gamma12=1,
        gamma21=1,
    )
    model.plot_surface_plotly(d1, d2, fname="parametric.html")
    assert 1 == 1


def test_parametric_model_heatmap_replicates():
    from synergy.utils.dose_tools import make_dose_grid
    from synergy.combination import MuSyC

    d1, d2 = make_dose_grid(0.01, 10, 0.01, 10, 5, 5, include_zero=True, replicates=3)
    model = MuSyC(
        E0=1,
        E1=0.2,
        E2=0.5,
        E3=0.1,
        h1=1.4,
        h2=0.9,
        C1=0.2,
        C2=0.2,
        alpha12=2,
        alpha21=0.5,
        gamma12=1,
        gamma21=1,
    )
    model.plot_heatmap(d1, d2, fname="parametric.pdf")
    assert 1 == 1

def test_heatmap_replicates():
    import numpy as np
    from synergy.datasets import sham
    from synergy.utils.plots import plot_heatmap

    d1, d2, E = sham()
    d1 = np.hstack([d1,d1])
    d2 = np.hstack([d2,d2])
    E = np.hstack([E,E*2])

    plot_heatmap(d1, d2, E, fname="x.pdf")
    assert 1==1

def test_heatmap_linearscale():
    import numpy as np
    from synergy.datasets import sham
    from synergy.utils.plots import plot_heatmap

    d1, d2, E = sham()
    d1 = np.hstack([d1,d1])
    d2 = np.hstack([d2,d2])
    E = np.hstack([E,E*2])

    plot_heatmap(d1, d2, E, fname="x.pdf", logscale=False)
    assert 1==1


def test_plotly_replicates():
    import numpy as np
    from synergy.datasets import sham
    from synergy.utils.plots import plot_surface_plotly

    d1, d2, E = sham()
    d1 = np.hstack([d1,d1])
    d2 = np.hstack([d2,d2])
    E = np.hstack([E,E*2])

    plot_surface_plotly(d1, d2, E, fname="x.html")
    assert 1==1

def test_plotly_linearscale():
    import numpy as np
    from synergy.datasets import sham
    from synergy.utils.plots import plot_surface_plotly

    d1, d2, E = sham()
    d1 = np.hstack([d1,d1])
    d2 = np.hstack([d2,d2])
    E = np.hstack([E,E*2])

    plot_surface_plotly(d1, d2, E, fname="x.html", logscale=False)
    assert 1==1
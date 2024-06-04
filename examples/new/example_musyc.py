"""-."""

from matplotlib import pyplot as plt

from synergy.combination.musyc import MuSyC
from synergy.utils import plots
from synergy.testing_utils.synthetic_data_generators import MuSyCDataGenerator
from synergy.plot.plot_mixins import PlotMixins

E0, E1, E2, E3 = 1, 0.2, 0.1, 0.4
h1, h2 = 2.3, 0.8
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 3.2, 1.1
gamma12, gamma21 = 1, 1

d1, d2, E = MuSyCDataGenerator.get_2drug_combination(
    E0=1, E1=0.2, E2=0.1, E3=0.4, h1=2.3, h2=0.8, C1=1e-2, C2=1e-1, alpha12=3.2, alpha21=1.1, n_points1=8, n_points2=12
)
model = MuSyC(fit_gamma=False)
model.fit(d1, d2, E)
print(model)

if False:
    fig = plt.figure(figsize=(7, 7))

    ax = fig.add_subplot(221)
    plots.plot_heatmap(d1, d2, E, ax=ax, title="Data", cmap="viridis")

    ax = fig.add_subplot(222)
    plots.plot_heatmap(d1, d2, model.E(d1, d2), ax=ax, title="Fit", cmap="viridis")

    # ax = fig.add_subplot(223)
    # model.plot_heatmap(d1, d2, ax=ax, title="Fit")

    # ax = fig.add_subplot(224)
    # model.plot_residual_heatmap(d1, d2, Efit, ax=ax, title="Residuals", center_on_zero=True)

    plt.tight_layout()
    plt.show()


PlotMixins.plot_heatmap_summary(model, d1, d2)

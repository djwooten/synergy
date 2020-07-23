from synergy.combination import MuSyC, CombinationIndex, BRAID, Zimmer
from synergy.utils.dose_tools import grid
from synergy.utils import plots
from matplotlib import pyplot as plt

mtrue = MuSyC(E0=1, E1=0.05, E2=0.75, E3=0.1, h1=0.8, h2=1, C1=5e1, C2=1e3, alpha12=1, alpha21=20, gamma21=0.5, gamma12=1)
model = CombinationIndex()

d1, d2 = grid(1, 1e4, 1., 1e4, 10, 10, include_zero=True)
E = mtrue.E(d1, d2)


model.fit(d1, d2, E)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(14,4)
fig.tight_layout(pad=3.0)

# Plot raw data heatmap
plots.plot_heatmap(d1, d2, E, ax=ax1, cmap="RdYlGn", vmin=-1, vmax=1)
ax1.set_title("Experimental Data")

# Plot Reference model
model.plot_reference_heatmap(ax=ax2, cmap="RdYlGn", vmin=-1, vmax=1)
ax2.set_title("CI Model")

# Plot synergy
model.plot_heatmap(ax=ax3, cmap="RdYlGn")
ax3.set_title("CI Synergy")
#plt.show()







model = MuSyC()
name = model.__class__.__name__
model.fit(d1, d2, E)

fig, axes = plt.subplots(nrows=2, ncols=3)
ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]
fig.set_size_inches(14,8)
fig.tight_layout(pad=3.0)

# Plot raw data
plots.plot_heatmap(d1, d2, E, ax=ax1, vmin=0, vmax=1, cmap="RdYlGn", title="Experimental Data")

# Plot model fit
model.plot_heatmap(d1, d2, ax=ax2, vmin=0, vmax=1, cmap="RdYlGn", title=f"{name} Fit")

# Plot differences between the raw data and the model fit. Ideally these
# values are small, and mostly random across all measurements.
model.plot_residual_heatmap(d1, d2, E, ax=ax3, title=f"{name} Fit Residuals")

# Skip ax4

# Plot model reference, which is obtained by setting all synergy parameters
# to their "additive" values
model.plot_reference_heatmap(d1, d2, ax=ax5, cmap="RdYlGn", title=f"{name} Reference")

# Plot "excess over reference" to get an idea of which doses show the
# biggest improvement above the zero-synergy version
model.plot_delta_heatmap(d1, d2, ax=ax6, title=f"{name} Delta")

plt.show()




def analyze_parametric(model, dfa, bootstrap_iterations=0):
    d1, d2, E = dfa['conc_c'], dfa['conc_r'], dfa['response']

    name = model.__class__.__name__
    model.fit(d1, d2, E, bootstrap_iterations=bootstrap_iterations)
    print(f"{name} Report:")

    print("\nParameters:")
    print(model.get_parameters())

    print("\nSummary:")
    print(model.summary())

    print(f'\nAverage Delta: {np.nanmean(model._reference_E(d1,d2)-model.E(d1,d2))}')

    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    fig.set_size_inches(14,8)
    fig.tight_layout(pad=3.0)

    # Plot raw data
    plots.plot_heatmap(d1, d2, E, ax=ax1, vmin=0, vmax=1, cmap="RdYlGn", title="Experimental Data")

    # Plot model fit
    model.plot_heatmap(d1, d2, ax=ax2, vmin=0, vmax=1, cmap="RdYlGn", title=f"{name} Fit")

    # Plot differences between the raw data and the model fit. Ideally these
    # values are small, and mostly random across all measurements.
    model.plot_residual_heatmap(d1, d2, E, ax=ax3, title=f"{name} Fit Residuals")

    # Skip ax4

    # Plot model reference, which is obtained by setting all synergy parameters
    # to their "additive" values
    model.plot_reference_heatmap(d1, d2, ax=ax5, cmap="RdYlGn", title=f"{name} Reference")

    # Plot "excess over reference" to get an idea of which doses show the
    # biggest improvement above the zero-synergy version
    model.plot_delta_heatmap(d1, d2, ax=ax6, title=f"{name} Delta")

    plt.show()
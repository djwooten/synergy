import numpy as np
from matplotlib import pyplot as plt

from synergy.combination.musyc import MuSyC
from synergy.combination.synergy_model_2d import ParametricSynergyModel2D, SynergyModel2D
from synergy.plot import constants
from synergy.utils import plots


class PlotMixins:
    """Mixins related to various synergy plots"""

    @staticmethod
    def _synergy_plot_transform(self):
        """-"""
        return self.synergy

    @staticmethod
    def plot_synergy_heatmap(model: SynergyModel2D, d1, d2, cmap=constants.CMAP_SYNERGY, title="Synergy", **kwargs):
        """-"""
        E = model.E(d1, d2)
        E_reference = model.E_reference(d1, d2)
        plots.plot_heatmap(d1, d2, E_reference - E, cmap=cmap, title=title, **kwargs)

    @staticmethod
    def plot_reference_heatmap(model: SynergyModel2D, d1, d2, cmap=constants.CMAP_E, title="Reference", **kwargs):
        """-"""
        E_reference = model.E_reference(d1, d2)
        plots.plot_heatmap(d1, d2, E_reference, cmap=cmap, title=title, **kwargs)

    @staticmethod
    def plot_model_heatmap(model: SynergyModel2D, d1, d2, cmap=constants.CMAP_E, title="Model", **kwargs):
        """-"""
        E = model.E(d1, d2)
        plots.plot_heatmap(d1, d2, E, cmap=cmap, title=title, **kwargs)

    @staticmethod
    def plot_synergy_parameters(model: MuSyC, ax, confidence_interval=95):  # noqa: F821
        """-"""
        parameters = model.get_parameters()
        parameters["beta"] = model.beta
        x = list(range(len(parameters)))
        heights = [parameters[i] for i in parameters]
        try:
            confidence_intervals = model.get_confidence_intervals(confidence_interval=confidence_interval)
            yerr = np.asarray([confidence_intervals[i] for i in parameters])
        except ValueError:
            yerr = None

        ax.bar(x, heights, yerr=yerr)

    @staticmethod
    def plot_heatmap_summary(model, d1, d2, figsize=(8, 6), **kwargs):
        """-"""
        plt.figure(figsize=figsize)
        if kwargs.pop("vmin", None):
            print("TODO LOGGER this overrides vmin")
        if kwargs.pop("vmax", None):
            print("TODO LOGGER this overrides vmax")
        ax_model = plt.subplot2grid((2, 3), (0, 0))
        ax_reference = plt.subplot2grid((2, 3), (1, 0))
        ax_synergy = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)

        E = model.E(d1, d2)
        E_reference = model.E_reference(d1, d2)
        all_E = np.hstack([E, E_reference])
        vmin = np.nanmin(all_E)
        vmax = np.nanmax(all_E)

        PlotMixins.plot_model_heatmap(model, d1, d2, ax=ax_model, vmin=vmin, vmax=vmax, **kwargs)
        PlotMixins.plot_reference_heatmap(model, d1, d2, ax=ax_reference, vmin=vmin, vmax=vmax, **kwargs)
        # PlotMixins.plot_synergy_heatmap(model, d1, d2, ax=ax_synergy, center_on_zero=True, **kwargs)
        PlotMixins.plot_synergy_parameters(model, ax=ax_synergy)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_synergy_surface(model: SynergyModel2D, d1, d2, cmap=constants.CMAP_E, **kwargs):
        """-"""


class PlotMixinsND:
    """-"""

    def __init__(self):
        self.plot_mixins_2d = PlotMixins()

    def plot_synergy_heatmap(self, d, drugs: list = None, cmap=constants.CMAP_SYNERGY, **kwargs):
        """-"""
        drugs = drugs or [0, 1]
        self.plot_mixins_2d(d[drugs[0]], d[drugs[1]], cmap=cmap, **kwargs)

    def junk(self):
        """-"""
        model = MagicMock()
        model.plot_synergy_heatmap()
        model.plot_reference_heatmap()
        model.plot_model_heatmap()

        model.plot_synergy_surface()
        model.plot_reference_surface()
        model.plot_model_surface()


class MagicMock:
    """-"""

"""Helper functions to make plots of drug combination responses and synergy."""

import logging
from typing import Sequence

import numpy as np

from synergy.utils.dose_utils import aggregate_replicates, is_on_grid, remove_zeros

SUPPORTED_PLOTLY_EXTENSIONS = ["png", "jpeg", "jpg", "webp", "svg", "pdf", "eps"]
_LOGGER = logging.Logger(__name__)

matplotlib_import = False
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    matplotlib_import = True

except ImportError:
    _LOGGER.warning("Some plotting functions will not work unless matplotlib is installed.")

plotly_import = False
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly import offline

    plotly_import = True

except ImportError:
    _LOGGER.warning("Some plotting functions will not work unless plotly is installed.")

pandas_import = False
try:
    import pandas as pd  # noqa: F401

    pandas_import = True

except ImportError:
    _LOGGER.warning("Some plotting functions will not work unless pandas is installed.")


_PLOTLY_PLOT_INTERACTIVE = False


def set_plotly_interactive(interactive=True):
    """Configures plotly to use iplot() instead of plot(), such as within a Jupyter notebook."""
    if not plotly_import:
        raise ImportError("plotly must be installed to set plotly to interactive mode")
    global _PLOTLY_PLOT_INTERACTIVE
    _PLOTLY_PLOT_INTERACTIVE = interactive
    # offline.init_notebook_mode(connected=interactive)


def _get_extension(fname):
    if "." not in fname:
        return ""
    return fname.split(".")[-1].lower()


def _get_cmap(**kwargs):
    """Return a colormap based on the kwargs.

    kwargs:
        cmap: str or matplotlib.colors.Colormap colormap to use
        nancolor: str color to use for NaN values (only if cmap is a string)
    """
    cmap = kwargs.pop("cmap", "PRGn")
    if isinstance(cmap, str):
        cmap = plt.get_cmap(name=cmap)
        cmap.set_bad(color=kwargs.pop("nancolor", "#BBBBBB"))
    return cmap


def _get_vmin_vmax(vals, vmin, vmax, center_on_zero):
    """Return vmin and vmax based on the kwargs.

    :param float vmin: minimum value for the color scale (or None to use the minimum value in vals)
    :param float vmax: maximum value for the color scale (or None to use the maximum value in vals)
    :param bool center_on_zero: if True, set vmin and vmax to symmetric values around zero
    """
    if center_on_zero:
        if vmin is None or vmax is None:
            if not (vmin is None and vmax is None):
                _LOGGER.warning(
                    f"center_on_zero=True expects vmin ({vmin}) and vmax ({vmax}) to both be None, or both be"
                    " specified. Ignoring the only specified value and using min and max vals instead."
                )
            zmax = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)))
            vmin = -zmax
            vmax = zmax
        else:
            zmax = max(abs(vmin), abs(vmax))
            vmin = -zmax
            vmax = zmax
    return vmin, vmax


def _get_ax(**kwargs):
    """Return an axis based on the kwargs.

    kwargs:
        ax: matplotlib axis or None to generate a new figure
        figsize: tuple of width and height
        aspect: str aspect ratio of the plot (default is "equal")

    Returns:
        Tuple[matplotlib axis, bool if the axis was created or already supplied]
    """
    ax = kwargs.pop("ax", None)
    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", None))
        ax = fig.add_subplot(111)
        created_ax = True
    ax.set_aspect(kwargs.pop("aspect", "equal"))
    return ax, created_ax


def _relabel_log_ticks(ax, d1, d2):
    """Relabel the x and y axes of a heatmap with log-scaled doses.

    :param ax: matplotlib axis
    :param d1: doses for the x-axis
    :param d2: doses for the y-axis
    """
    nx = len(d1)
    ny = len(d2)

    MIN_logx = np.log10(min(d1))
    MAX_logx = np.log10(max(d1))
    min_logx = int(np.ceil(np.log10(min(d1))))
    max_logx = int(np.floor(np.log10(max(d1))))

    MIN_logy = np.log10(min(d2))
    MAX_logy = np.log10(max(d2))
    min_logy = int(np.ceil(np.log10(min(d2))))
    max_logy = int(np.floor(np.log10(max(d2))))

    doses = np.arange(min_logx, max_logx + 1, 1)
    ticks = np.interp(doses, [MIN_logx, MAX_logx], [0.5, nx - 0.5])
    ticklabels = [r"$10^{{{}}}$".format(dose) for dose in doses]

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    minor_ticks = []
    for i in range(min_logx - 1, max_logx + 1):
        for j in range(2, 10):
            minor_ticks.append(i + np.log10(j))
    minor_ticks = _interp(minor_ticks, MIN_logx, MAX_logx, 0.5, nx - 0.5)
    minor_ticks = [i for i in minor_ticks if i > 0 and i < nx]

    ax.set_xticks(minor_ticks, minor=True)

    doses = np.arange(min_logy, max_logy + 1, 1)
    ticks = np.interp(doses, [MIN_logy, MAX_logy], [0.5, ny - 0.5])
    ticklabels = [r"$10^{{{}}}$".format(dose) for dose in doses]

    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    minor_ticks = []
    for i in range(min_logy - 1, max_logy + 1):
        for j in range(2, 10):
            minor_ticks.append(i + np.log10(j))
    minor_ticks = _interp(minor_ticks, MIN_logy, MAX_logy, 0.5, ny - 0.5)
    minor_ticks = [i for i in minor_ticks if i > 0 and i < ny]

    ax.set_yticks(minor_ticks, minor=True)


def _interp(x, x0: float, x1: float, y0: float, y1: float):
    """Interpolate values of x between x0 and x1 to y values between y0 and y1.

    :param ArrayLike x: array of values to interpolate
    :param x0: minimum of x
    :param x1: maximum of x
    :param y0: minimum of y
    :param y1: maximum of y
    :return: array of interpolated values
    """
    return (np.asarray(x) - x0) * (y1 - y0) / (x1 - x0) + y0


def plot_heatmap(
    d1,
    d2,
    vals,
    title: str = "",
    xlabel: str = "Drug 1",
    ylabel: str = "Drug 2",
    fname: str = "",
    **kwargs,
):
    """Plot a heatmap of drug combination data.

    This may be the raw response, dose-dependent synergy scores, model residuals, or any other data that can be
    represented as a grid of dose-dependent values.

    :param ArrayLike d1: array of doses for drug 1
    :param ArrayLike d2: array of doses for drug 2
    :param ArrayLike vals: array of effect values
    :param str title: title of the plot
    :param str xlabel: label for the x-axis (e.g., drug name, concentration units)
    :param str ylabel: label for the y-axis
    :param str fname: filename to save the plot (if not empty)
    :param kwargs: additional keyword arguments to configure the plot

        - aggfunc: Callable function to aggregate replicates (default is np.median)
        - aspect: str aspect ratio of the plot (default is "equal")
        - ax: matplotlib axis or None to generate a new figure
        - figsize: tuple of width and height for the figure
        - logscale: bool if True, plot the doses on a log scale
        - cmap: str or matplotlib.colors.Colormap colormap to use
        - nancolor: str color to use for NaN values (only if cmap is a string)
        - vmin: float minimum value for the color scale (or None to use the minimum value in vals)
        - vmax: float maximum value for the color scale (or None to use the maximum value in vals)
        - center_on_zero: bool if True, set vmin and vmax to symmetric values around zero
    """
    if not matplotlib_import:
        raise ImportError("matplotlib must be installed to plot")

    logscale = kwargs.pop("logscale", True)
    if logscale:
        d1 = remove_zeros(d1)
        d2 = remove_zeros(d2)
    else:
        d1 = np.array(d1, copy=True)
        d2 = np.array(d2, copy=True)
    vals = np.asarray(vals)
    sorted_indices = np.lexsort((d1, d2))
    D1 = d1[sorted_indices]
    D2 = d2[sorted_indices]
    vals = vals[sorted_indices]

    # Replicates
    D_unique, vals = aggregate_replicates(np.vstack((D1, D2)).T, vals, aggfunc=kwargs.pop("aggfunc", np.median))
    if not is_on_grid(D_unique):
        raise ValueError("plot_heatmap() requires d1, d2 to represent a dose grid")

    D1 = D_unique[:, 0]
    D2 = D_unique[:, 1]

    n_d1 = len(np.unique(D1))
    n_d2 = len(np.unique(D2))

    ax, created_ax = _get_ax(**kwargs)

    vmin, vmax = _get_vmin_vmax(
        vals, kwargs.pop("vmin", None), kwargs.pop("vmax", None), kwargs.pop("center_on_zero", False)
    )
    cmap = _get_cmap(**kwargs)

    if not logscale:
        D1, D2 = np.meshgrid(D1, D2)
        pco = ax.pcolormesh(D1, D2, vals.reshape(n_d2, n_d1), vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        pco = ax.pcolormesh(vals.reshape(n_d2, n_d1), cmap=cmap, vmin=vmin, vmax=vmax)
        _relabel_log_ticks(ax, np.unique(D1), np.unique(D2))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=max(2 / n_d1, 2 / n_d2, 0.05), pad=0.1)
    plt.colorbar(pco, cax=cax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if fname:
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    elif created_ax:
        plt.tight_layout()
        plt.show()


def plot_surface_plotly(
    d1,
    d2,
    vals,
    scatter_points=None,
    logscale: bool = True,
    xlabel: str = "Drug 1",
    ylabel: str = "Drug 2",
    zlabel: str = "z",
    title: str = "",
    fname: str = "",
    **kwargs,
):
    """Plot 3d surface of drug combination data.

    :param ArrayLike d1: array of doses for drug 1
    :param ArrayLike d2: array of doses for drug 2
    :param ArrayLike vals: array of values
    :param scatter_points: pandas dataframe of points to scatter on the surface plot
    :param bool logscale: if True, plot the doses on a log scale
    :param str xlabel: label for the x-axis (e.g., drug name, concentration units)
    :param str ylabel: label for the y-axis
    :param str zlabel: label for the z-axis
    :param str title: title of the plot
    :param str fname: filename to save the plot (if not empty)
    :param kwargs: additional keyword arguments to configure the plot

        - figsize: tuple of width and height for the figure
        - font: dict of font properties
        - fontsize: int font size (if font is not specified)
        - cmap: colormap str
        - vmin: float minimum value for the color scale (or None to use the minimum value in vals)
        - vmax: float maximum value for the color scale (or None to use the maximum value in vals)
        - center_on_zero: bool if True, set vmin and vmax to symmetric values around
    """
    if not plotly_import:
        raise ImportError("plot_surface_plotly() requires plotly to be installed.")

    d1 = np.array(d1, copy=True, dtype=np.float64)
    d2 = np.array(d2, copy=True, dtype=np.float64)
    vals = np.asarray(vals)

    if logscale:
        d1 = remove_zeros(d1)
        d2 = remove_zeros(d2)
        d1 = np.log10(d1)
        d2 = np.log10(d2)

    sorted_indices = np.lexsort((d1, d2))
    D1 = d1[sorted_indices]
    D2 = d2[sorted_indices]
    vals = vals[sorted_indices]

    # Replicates
    D_unique, vals = aggregate_replicates(np.vstack((D1, D2)).T, vals, aggfunc=kwargs.pop("aggfunc", np.median))
    if not is_on_grid(D_unique):
        raise ValueError("plot_surface_plotly() requires d1, d2 to represent a dose grid")

    D1 = D_unique[:, 0]
    D2 = D_unique[:, 1]

    n_d1 = len(np.unique(D1))
    n_d2 = len(np.unique(D2))

    vals = vals.reshape(n_d2, n_d1)
    d1 = D1.reshape(n_d2, n_d1)
    d2 = D2.reshape(n_d2, n_d1)

    if not title and fname:
        title = fname

    vmin, vmax = _get_vmin_vmax(
        vals, kwargs.pop("vmin", None), kwargs.pop("vmax", None), kwargs.pop("center_on_zero", False)
    )
    font = kwargs.pop("font", dict(size=kwargs.pop("fontsize", 18)))
    width, height = kwargs.pop("figsize", (1000, 800))

    if "opacity" not in kwargs:
        kwargs["opacity"] = 0.8
    if "contours_z" not in kwargs:
        kwargs["contours_z"] = dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=False)
    if "colorscale" not in kwargs:
        kwargs["colorscale"] = kwargs.pop("cmap", "PRGn")

    data_to_plot = [
        go.Surface(
            x=d1,
            y=d2,
            z=vals,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(lenmode="fraction", len=0.65, title=zlabel),
            **kwargs,
        ),
    ]

    if scatter_points is not None:
        d1scatter = np.array(scatter_points["drug1.conc"], copy=True, dtype=np.float64)
        d2scatter = np.array(scatter_points["drug2.conc"], copy=True, dtype=np.float64)
        if logscale:
            d1scatter = np.log10(remove_zeros(d1scatter))
            d2scatter = np.log10(remove_zeros(d2scatter))

        data_to_plot.append(
            go.Scatter3d(
                x=d1scatter,
                y=d2scatter,
                z=scatter_points["effect"],
                mode="markers",
                marker=dict(
                    size=3.0,
                    color=scatter_points["effect"],
                    colorscale=kwargs["colorscale"],
                    reversescale=kwargs.get("reversescale", False),
                    cmin=vmin,
                    cmax=vmax,
                    line={"width": 0.5, "color": "DarkSlateGrey"},
                ),
            )
        )

    fig = go.Figure(data=data_to_plot)

    fig.update_layout(
        title=title,
        autosize=False,
        scene_camera_eye=dict(x=1.87, y=0.88, z=0.64),
        width=width,
        height=height,
        margin=dict(l=100, r=100, b=90, t=90),
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel, aspectmode="cube"),
        font=font,
    )

    zlim = kwargs.pop("zlim", None)
    if zlim is not None:
        fig.update_layout(
            scene=dict(
                zaxis=dict(
                    range=zlim,
                )
            )
        )

    if fname:
        extension = _get_extension(fname)
        if extension == "html":
            offline.plot(fig, filename=fname, auto_open=False)
        else:
            if extension not in SUPPORTED_PLOTLY_EXTENSIONS:
                raise ValueError(
                    f"Extension {extension} is not supported. Supported extensions are {SUPPORTED_PLOTLY_EXTENSIONS}"
                )
            pio.write_image(fig, fname, format=extension)
    else:
        if _PLOTLY_PLOT_INTERACTIVE:
            offline.iplot(fig)
        else:
            fig.show()


def plotly_isosurfaces(
    d,
    vals,
    drug_indices: Sequence[int] = [0, 1, 2],
    fname: str = "",
    xlabel: str = "Drug 1",
    ylabel: str = "Drug 2",
    zlabel: str = "Drug 3",
    logscale: bool = True,
    surface_count: int = 10,
    title: str = "",
    **kwargs,
):
    """Plot isosurfaces of drug combination data.

    :param ArrayLike d: array of doses for each drug
    :param ArrayLike E: array of effect values
    :param Sequence[int] drug_indices: indices of the drugs to plot
    :param str fname: filename to save the plot (if not empty)
    :param str xlabel: label for the x-axis (e.g., drug name, concentration units)
    :param str ylabel: label for the y-axis
    :param str zlabel: label for the z-axis
    :param bool logscale: if True, plot the doses on a log scale
    :param int surface_count: number of isosurfaces
    :param str title: title of the plot
    :param kwargs: additional keyword arguments to configure the plot

        - figsize: tuple of width and height for the figure
        - font: dict of font properties
        - fontsize: int font size (if font is not specified)
        - cmap: colormap str
        - vmin: float minimum value for the color scale (or None to use the minimum value in vals)
        - vmax: float maximum value for the color scale (or None to use the maximum value in vals)
        - center_on_zero: bool if True, set vmin and vmax to symmetric values around
        - isomin: float minimum value for the isosurfaces
        - isomax: float maximum value for the isosurfaces
    """
    if d.shape[1] < 3:
        raise ValueError(f"plotly_isosurfaces() requires at least 3 drugs to plot. d.shape[1] == {d.shape[1]} (< 3).")
    if len(drug_indices) != 3:
        raise ValueError(
            f"plotly_isosurfaces() requires exactly 3 drug indices. len(drug_indices) == {len(drug_indices)}"
        )
    for drug_index in range(d.shape[1]):
        if drug_index in drug_indices:
            continue
        d_unique = np.unique(d[:, drug_index])
        if len(d_unique) > 1:
            raise ValueError(
                f"All drugs except those specified in drug_indices {drug_indices} are expected to be at a constant"
                " slice. Drug {drug_index} has more than one unique value."
            )
    d, vals = aggregate_replicates(d, vals)
    d1 = d[:, drug_indices[0]]
    d2 = d[:, drug_indices[1]]
    d3 = d[:, drug_indices[2]]

    if logscale:
        d1 = remove_zeros(d1)
        d2 = remove_zeros(d2)
        d3 = remove_zeros(d3)
        d1 = np.log10(d1)
        d2 = np.log10(d2)
        d3 = np.log10(d3)

    vmin, vmax = _get_vmin_vmax(
        vals, kwargs.pop("vmin", None), kwargs.pop("vmax", None), kwargs.pop("center_on_zero", False)
    )
    width, height = kwargs.pop("figsize", (1000, 800))
    font = kwargs.pop("font", dict(size=kwargs.pop("fontsize", 18)))

    if "colorscale" not in kwargs:
        kwargs["colorscale"] = kwargs.pop("cmap", "Viridis")

    isomin = kwargs.pop("isomin", None)
    isomax = kwargs.pop("isomax", None)
    E_range = np.nanmax(vals[~np.isinf(vals)]) - np.nanmin(vals[~np.isinf(vals)])
    if isomin is None:
        isomin = np.nanmin(vals[~np.isinf(vals)]) + 0.1 * E_range
    if isomax is None:
        isomax = np.nanmin(vals[~np.isinf(vals)]) + 0.9 * E_range

    fig = go.Figure(
        data=go.Isosurface(
            x=d1,
            y=d2,
            z=d3,
            value=vals,
            isomin=isomin,
            isomax=isomax,
            cmin=vmin,
            cmax=vmax,
            surface_count=surface_count,  # number of isosurfaces, 2 by default: only min and max
            colorbar_nticks=surface_count,  # colorbar ticks correspond to isosurface values
            caps=dict(x_show=False, y_show=False, z_show=True),
            **kwargs,
        )
    )

    if not title and fname:
        title = fname

    fig.update_layout(
        title=title,
        autosize=False,
        scene_camera_eye=dict(x=1.87, y=0.88, z=0.64),
        width=width,
        height=height,
        margin=dict(l=100, r=100, b=90, t=90),
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel, aspectmode="cube"),
        font=font,
    )

    if fname:
        extension = _get_extension(fname)
        if extension == "html":
            offline.plot(fig, filename=fname, auto_open=False)
        else:
            if extension not in SUPPORTED_PLOTLY_EXTENSIONS:
                raise ValueError(
                    f"Extension {extension} is not supported. Supported extensions are {SUPPORTED_PLOTLY_EXTENSIONS}"
                )
            pio.write_image(fig, fname, format=extension)
    else:
        if _PLOTLY_PLOT_INTERACTIVE:
            offline.iplot(fig)
        else:
            fig.show()

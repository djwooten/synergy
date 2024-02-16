#    Copyright (C) 2020 David J. Wooten
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from . import base as utils
from .dose_tools import get_num_replicates

matplotlib_import = False
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.cm import get_cmap
    
    matplotlib_import = True
    
except ImportError as e:
    pass

plotly_import = False
try:
    import plotly.graph_objects as go
    from plotly import offline
    plotly_import = True
except ImportError as e:
    pass

def get_extension(fname):
    if not "." in fname: return ""
    return fname.split(".")[-1]

def plot_heatmap(d1, d2, E, ax=None, fname=None, title="", xlabel="Drug 1", \
                 ylabel="Drug 2", figsize=None, cmap="PRGn", aspect='equal', \
                 vmin=None, vmax=None, center_on_zero=False, logscale=True, \
                 nancolor="#BBBBBB", **kwargs):
    if (not matplotlib_import):
        raise ImportError("matplotlib must be installed to plot")
    
    if logscale:
        d1 = utils.remove_zeros(d1)
        d2 = utils.remove_zeros(d2)
    else:
        d1 = np.array(d1, copy=True)
        d2 = np.array(d2, copy=True)
    E = np.asarray(E)
    sorted_indices = np.lexsort((d1,d2))
    D1 = d1[sorted_indices]
    D2 = d2[sorted_indices]
    E = E[sorted_indices]
    
    # Replicates
    n_replicates = np.unique(get_num_replicates(d1,d2))
    if len(n_replicates)>1:
        raise ValueError("plot_heatmap() expects the same number of replicates for each dose")
    n_replicates = n_replicates[0]
    
    if n_replicates != 1:
        aggfunc = kwargs.get('aggfunc', np.median)
        print(F'Number of replicates : {n_replicates}. plot_heatmap() aggregates the data using the {aggfunc.__name__}')
        E_agg = []
        for e2 in np.unique(D2):
            for e1 in np.unique(D1):
                ix = (D1==e1) & (D2==e2)
                E_agg.append(aggfunc(E[ix]))
        E = np.array(E_agg)
        D1 = np.unique(D1)
        D2 = np.unique(D2)
        title += (F'({aggfunc.__name__})')

    n_d1 = len(np.unique(D1))
    n_d2 = len(np.unique(D2))

    if len(d1) != n_d1*n_d2*n_replicates :
        raise ValueError("plot_heatmap() requires d1, d2 to represent a dose grid")
    
    
    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        created_ax=True


    if center_on_zero:
        if vmin is None or vmax is None:
            zmax = max(abs(np.nanmin(E)), abs(np.nanmax(E)))
            vmin = -zmax
            vmax = zmax
        else:
            zmax = max(abs(vmin), abs(vmax))
            vmin = -zmax
            vmax = zmax


    current_cmap = get_cmap(name=cmap)
    current_cmap.set_bad(color=nancolor)

    if not logscale:
        D1, D2 = np.meshgrid(D1, D2)
        #pco = ax.pcolormesh(D1.reshape(n_d2,n_d1), D2.reshape(n_d2, n_d1), E.reshape(n_d2, n_d1), vmin=vmin, vmax=vmax, cmap=cmap)
        pco = ax.pcolormesh(D1, D2, E.reshape(n_d2, n_d1), vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        pco = ax.pcolormesh(E.reshape(n_d2, n_d1), cmap=cmap, vmin=vmin, vmax=vmax)
        relabel_log_ticks(ax, np.unique(D1), np.unique(D2))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=max(2/n_d1, 2/n_d2, 0.05), pad=0.1)
    plt.colorbar(pco, cax=cax)

    ax.set_aspect(aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if (fname is not None):
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    elif (created_ax):
        plt.tight_layout()
        plt.show()



def square_log_axes(ax, nx, ny):
    """
    Transform log-scaled axes into a square aspect ratio.
    The axes will be resized so that each cell in the heatmap is square.
    TODO: This will work, but only if tight_layout() is not called
    """
    ratio = ny / nx

    pos1 = ax.get_position()

    fig_width = ax.get_figure().get_figwidth()
    fig_height = ax.get_figure().get_figheight()
    axratio = (fig_height*pos1.height)/(fig_width*pos1.width)
    
    pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height]   

    if (axratio > ratio):
        # decrease height
        target_height = ratio*fig_width*pos1.width/fig_height
        delta_height = pos1.height - target_height
        pos2[1] = pos2[1]-delta_height/2. # Move y0 up by delta/2
        pos2[3] = target_height
        ax.set_position(pos2)

    elif (axratio < ratio):
        # decrease width
        target_width = fig_height*pos1.height/(fig_width*ratio)
        #target_width = pos1.width/10
        delta_width = pos1.width - target_width
        pos2[0] = pos2[0]+delta_width/2. # Move x0 right by delta/2
        pos2[2] = target_width
        #pos2 = [0,0,1,1]
        ax.set_position(pos2)

def relabel_log_ticks(ax, d1, d2):
    """
    In plotting using pcolormesh(E), the x and y axes go from 0 to nx (or ny).
    This function replaces those with tick marks reflecting the true doses.
    Assumes both x and y axes come from log-scaled doses
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

    doses = np.arange(min_logx, max_logx+1, 1)
    ticks = np.interp(doses,[MIN_logx,MAX_logx],[0.5,nx-0.5]) 
    ticklabels = [r"$10^{{{}}}$".format(dose) for dose in doses]

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    minor_ticks = []
    for i in range(min_logx-1, max_logx+1):
        for j in range(2,10):
            minor_ticks.append(i+np.log10(j))
    minor_ticks = interp(minor_ticks,MIN_logx,MAX_logx,0.5,nx-0.5)
    minor_ticks = [i for i in minor_ticks if i>0 and i<nx]

    ax.set_xticks(minor_ticks, minor=True)




    doses = np.arange(min_logy, max_logy+1, 1)
    ticks = np.interp(doses,[MIN_logy,MAX_logy],[0.5,ny-0.5]) 
    ticklabels = [r"$10^{{{}}}$".format(dose) for dose in doses]

    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)


    minor_ticks = []
    for i in range(min_logy-1, max_logy+1):
        for j in range(2,10):
            minor_ticks.append(i+np.log10(j))
    minor_ticks = interp(minor_ticks,MIN_logy,MAX_logy,0.5,ny-0.5)
    minor_ticks = [i for i in minor_ticks if i>0 and i<ny]

    ax.set_yticks(minor_ticks, minor=True)

def interp(x, x0, x1, y0, y1):
    return (np.asarray(x)-x0)*(y1-y0)/(x1-x0)+y0

def plot_surface_plotly(d1, d2, E, scatter_points=None, elev=20, azim=19, \
                 fname="plot.html", zlim=None, cmap='viridis', logscale=True, \
                 xlabel="Drug 1", ylabel="Drug 2", zlabel="z", \
                 vmin=None, vmax=None, auto_open=False, opacity=0.8, \
                 center_on_zero=False, figsize=(1000,800), \
                 fontsize=18, title=None, **kwargs):
    if (not plotly_import):
        raise ImportError("plot_surface_plotly() requires plotly to be installed.")
    
    d1 = np.array(d1, copy=True, dtype=np.float64)
    d2 = np.array(d2, copy=True, dtype=np.float64)
    E = np.asarray(E)

    extension = get_extension(fname)

    if logscale:
        d1 = utils.remove_zeros(d1)
        d2 = utils.remove_zeros(d2)
        d1 = np.log10(d1)
        d2 = np.log10(d2)

    sorted_indices = np.lexsort((d1,d2))
    D1 = d1[sorted_indices]
    D2 = d2[sorted_indices]
    E = E[sorted_indices]

    if title is None:
        if fname is not None:
            title = fname
        else:
            title = ""

    # Replicates
    n_replicates = np.unique(get_num_replicates(D1,D2))
    if len(n_replicates)>1:
        raise ValueError("plot_surface_plotly() expects the same number of replicates for each dose")
    n_replicates = n_replicates[0]
    
    if n_replicates != 1:
        aggfunc = kwargs.get('aggfunc', np.median)
        print(F'Number of replicates : {n_replicates}. plot_surface_plotly() aggregates the data using the {aggfunc.__name__}')
        E_agg = []
        for e2 in np.unique(D2):
            for e1 in np.unique(D1):
                ix = (D1==e1) & (D2==e2)
                E_agg.append(aggfunc(E[ix]))
        E = np.array(E_agg)
        D1 = np.unique(D1)
        D2 = np.unique(D2)
        title += (F'({aggfunc.__name__})')

        d1, d2 = np.meshgrid(D1, D2)

        n_d1 = len(np.unique(D1))
        n_d2 = len(np.unique(D2))
        E = E.reshape(n_d2,n_d1)
    else:
        n_d1 = len(np.unique(d1))
        n_d2 = len(np.unique(d2))
        d1 = d1.reshape(n_d2,n_d1)
        d2 = d2.reshape(n_d2,n_d1)
        E = E.reshape(n_d2,n_d1)

    if center_on_zero:
        if vmin is None or vmax is None:
            zmax = max(abs(np.nanmin(E[~np.isinf(E)])), abs(np.nanmax(E[~np.isinf(E)])))
            vmin = -zmax
            vmax = zmax
        else:
            zmax = max(abs(vmin), abs(vmax))
            vmin = -zmax
            vmax = zmax

    data_to_plot = [go.Surface(
        x=d1,
        y=d2,
        z=E,
        cmin=vmin,
        cmax=vmax,
        opacity=opacity,
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=False
        ),
        colorscale=cmap,
        reversescale=False,
        colorbar=dict(
            lenmode='fraction',
            len=0.65,
            title=zlabel
            )
        ),
    ]

    if scatter_points is not None:
        #d1scatter = utils.remove_zeros(np.asarray(scatter_points['drug1.conc']))
        #d2scatter = utils.remove_zeros(np.asarray(scatter_points['drug2.conc']))
        d1scatter = np.array(scatter_points['drug1.conc'], copy=True, dtype=np.float64)
        d2scatter = np.array(scatter_points['drug2.conc'], copy=True, dtype=np.float64)
        if logscale:
            zero_mask_1 = np.where(d1scatter <= 0)
            pos_mask_1 = np.where(d1scatter > 0)

            zero_mask_2 = np.where(d2scatter <= 0)
            pos_mask_2 = np.where(d2scatter > 0)

            d1scatter[zero_mask_1] = np.min(d1)
            d2scatter[zero_mask_2] = np.min(d2)
            d1scatter[pos_mask_1] = np.log10(d1scatter[pos_mask_1])
            d2scatter[pos_mask_2] = np.log10(d2scatter[pos_mask_2])
        
        data_to_plot.append(go.Scatter3d(
            x=d1scatter,
            y=d2scatter,
            z=scatter_points['effect'],
            mode="markers",
            marker=dict(
                size=3.0,
                color=scatter_points['effect'],
                colorscale=cmap,
                reversescale=False,
                cmin=vmin,
                cmax=vmax,
                line=dict(
                    width=0.5,
                    color='DarkSlateGrey'
                )
            )
        ))

    fig = go.Figure(data=data_to_plot)

    #fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

    fig.update_layout(
        title=title,
        autosize=False,
        scene_camera_eye=dict(
            x=1.87, 
            y=0.88, 
            z=0.64
        ), 
        width=figsize[0], 
        height=figsize[1], 
        margin=dict(
            l=100, 
            r=100, 
            b=90, 
            t=90
        ), 
        scene=dict(
            xaxis_title=xlabel, 
            yaxis_title=ylabel, 
            zaxis_title=zlabel, 
            aspectmode="cube"
        ),
        font=dict(
            size=fontsize
        )
    )

    if zlim is not None:
        fig.update_layout(scene = dict(zaxis = dict(range=zlim,)))


    if fname is not None:
        if extension=="html":
            offline.plot(fig, filename=fname, auto_open=auto_open)
        else:
            if not extension.lower() in ['png','jpeg','jpg','webp','svg','pdf','eps']:
                extension='png'
            from plotly.io import write_image
            write_image(fig, fname, format=extension)
    else:
        fig.show()


def plotly_isosurfaces(d1, d2, d3, E, fname=None,     \
            cmap='viridis', xlabel="Drug 1", ylabel="Drug 2", \
            zlabel="Drug 3",     \
            vmin=None, vmax=None, auto_open=False, opacity=0.6,      \
            logscale=True, isomin=None, isomax=None,                \
            center_on_zero=False, surface_count=10, \
            figsize=(1000,800), fontsize=18, title=None):
    
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    d3 = np.asarray(d3)
    
    if (len(d1.shape) > 1):
        d1 = d1.flatten()
        d2 = d2.flatten()
        d3 = d3.flatten()
        E = E.flatten()

    sorted_indices = np.lexsort((d1,d2,d3))
    d1 = d1[sorted_indices]
    d2 = d2[sorted_indices]
    d3 = d3[sorted_indices]
    E = E[sorted_indices]

    if logscale:
        d1 = utils.remove_zeros(d1)
        d2 = utils.remove_zeros(d2)
        d3 = utils.remove_zeros(d3)
        d1 = np.log10(d1)
        d2 = np.log10(d2)
        d3 = np.log10(d3)
    
    
    E_range = np.nanmax(E[~np.isinf(E)]) - np.nanmin(E[~np.isinf(E)])
    if isomin is None:
        isomin = np.nanmin(E[~np.isinf(E)]) + 0.1*E_range
    if isomax is None:
        isomax = np.nanmin(E[~np.isinf(E)]) + 0.9*E_range

    if center_on_zero:
        if vmin is None or vmax is None:
            zmax = max(abs(np.nanmin(E[~np.isinf(E)])), abs(np.nanmax(E[~np.isinf(E)])))
            vmin = -zmax
            vmax = zmax
        else:
            zmax = max(abs(vmin), abs(vmax))
            vmin = -zmax
            vmax = zmax

        
    fig = go.Figure(data=go.Isosurface(
        x=d1,
        y=d2,
        z=d3,
        value=E,
        isomin=isomin,
        isomax=isomax,
        cmin=vmin,
        cmax=vmax,
        opacity=0.6,
        colorscale=cmap,
        surface_count=surface_count, # number of isosurfaces, 2 by default: only min and max
        colorbar_nticks=surface_count, # colorbar ticks correspond to isosurface values
        caps=dict(x_show=False, y_show=False, z_show=True)
    ))

    if title is None:
        if fname is not None:
            title = fname
        else:
            title = ""

    fig.update_layout(
        title=title,
        autosize=False,
        scene_camera_eye=dict(
            x=1.87, 
            y=0.88, 
            z=0.64
        ), 
        width=figsize[0],
        height=figsize[1],
        margin=dict(
            l=100, 
            r=100, 
            b=90, 
            t=90
        ), 
        scene=dict(
            xaxis_title=xlabel, 
            yaxis_title=ylabel, 
            zaxis_title=zlabel, 
            aspectmode="cube"
        ),
        font=dict(
            size=fontsize
        )
    )

    if fname is not None:
        offline.plot(fig, filename=fname, auto_open=auto_open)
    else:
        fig.show()

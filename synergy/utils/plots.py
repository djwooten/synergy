import numpy as np

matplotlib_import = False
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    matplotlib_import = True
except ImportError as e:
    pass

pandas_import = False
try:
    import pandas as pd
    pandas_import = True
except ImportError as e:
    pass

def plot_colormap(d1, d2, E, ax=None, fname=None, title="", xlabel="", ylabel="", figsize=None, cmap=None, aspect='equal', **kwargs):
        if (not matplotlib_import):
            raise ImportError("matplotlib must be installed to plot")
        if (not pandas_import):
            raise ImportError("pandas must be installed to plot")
        
        df = pd.DataFrame(dict({'d1':d1, 'd2':d2, 'E':E}))
        df.sort_values(by=['d2','d1'],ascending=True, inplace=True)
        
        D1 = df['d1']
        D2 = df['d2']
        E = df['E']

        n_d1 = len(D1.unique())
        n_d2 = len(D2.unique())

        if len(d1) != n_d1*n_d2:
            raise ValueError("plot_colormap() requires d1, d2 to represent a dose grid")
        
        created_ax = False
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            created_ax=True

        #ax.pcolormesh(np.log10(D1.values.reshape(n_d2,n_d1)), np.log10(D2.values.reshape(n_d2, n_d1)), E.values.reshape(n_d2, n_d1))
        pco = ax.pcolormesh(E.values.reshape(n_d2, n_d1), cmap=cmap)
        #plt.colorbar(pco)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=max(2/n_d1, 2/n_d2, 0.05), pad=0.1)
        plt.colorbar(pco, cax=cax)
    
        #square_log_axes(ax, n_d1, n_d2)
        ax.set_aspect(aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        relabel_log_ticks(ax, D1.unique(), D2.unique())
        
        if (fname is not None):
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        elif (created_ax):
            plt.tight_layout()
            plt.show()



def square_log_axes(ax, nx, ny):
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
        print("Decreasing width")
        target_width = fig_height*pos1.height/(fig_width*ratio)
        #target_width = pos1.width/10
        delta_width = pos1.width - target_width
        pos2[0] = pos2[0]+delta_width/2. # Move x0 right by delta/2
        pos2[2] = target_width
        #pos2 = [0,0,1,1]
        ax.set_position(pos2)
        print(ratio, axratio, (pos1.x0, pos1.y0, pos1.width, pos1.height), pos2)

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def setup_canvas(nx, ny, figsize = (10, 6), sharex = True, sharey = True, markersize = 2, fontsize = 16, flatten = True, labelsize= 15, wspace = 0, hspace = 0, panels = False):
    plt.rcParams.update({'lines.markersize': markersize, 'font.size': fontsize})
    fig, axes = plt.subplots(nx, ny, figsize = figsize, sharex = sharex, sharey = sharey)
    if nx * ny == 1: axes = np.array([axes])
    if flatten: axes = axes.flatten()
    for ax in axes.flatten():
        ax.tick_params(direction = "in", which = "both", labelsize = labelsize)

    if panels:
        for i in range(len(axes)):
            axes[i].text(0.05, 0.8, f'({chr(97 + i)})', transform = axes[i].transAxes)

    plt.subplots_adjust(wspace = wspace, hspace = hspace)

    if len(axes) == 1:
        return fig, axes[0]
    else:
        return fig, axes

def upper_legend(ax, xloc = 0.5, yloc = 1.1, ncols = None, nrows = None, user_labels = [], user_labels_order = [], loc = "upper center", framealpha = 0., frameon = False):
    def reorder(list_in, nrows):
        ncols = len(list_in) // nrows
        if nrows * ncols != len(list_in): ncols += 1
        list_out = []
        for c in range(ncols):
            for r in range(nrows):
                if r * ncols + c >= len(list_in): continue
                list_out.append(list_in[r * ncols + c])
        assert len(list_in) == len(list_out), 'ERROR: len in != len out'
        return list_out, ncols
    handles, labels = ax.get_legend_handles_labels()
    if user_labels: labels = user_labels
    if user_labels_order:
        handles = [dict(zip(labels, handles))[l] for l in user_labels_order]
        labels = user_labels_order
    if len(handles) != len(labels): print('WARNING: the lengths are unequal')
    if nrows:
        labels, ncols = reorder(labels, nrows)
        handles, ncols = reorder(handles, nrows)
    by_label = dict(zip(labels, handles))
    if not ncols: ncols = len(labels)

    x0 = ax.get_position().x0
    x1 = ax.get_position().x1
    y0 = ax.get_position().y0
    y1 = ax.get_position().y1

    if xloc and yloc:
        ax.legend(by_label.values(), by_label.keys(), loc = loc, framealpha = framealpha, frameon = frameon, bbox_to_anchor=(xloc, yloc), ncol=ncols)
    else:
        ax.legend(by_label.values(), by_label.keys(), loc = loc, framealpha = framealpha, frameon = frameon, ncol=ncols, bbox_to_anchor=(0., -0.05, 1., 0.), borderaxespad=0, mode='expand')
    return ax

def nticks_prune(ax, which = 'x', nbins = None, prune = None):
    # prune: can be upper
    if which == 'x':
        if not nbins:
            nbins = len(ax.get_xticklabels()) # added 
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins = nbins, prune = prune))
    else:
        if not nbins:
            nbins = len(ax.get_yticklabels()) # added 
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins = nbins, prune = prune)) 
    return ax

def rotate_ticks(ax, which, degree):
    ax.tick_params(axis=which, rotation=degree)

def sort_list_by(lista, listb):
    # sort lista by listb
    lista = [i for _, i in sorted(zip(listb, lista))]
    return lista

def add_text(ax, x, y, text, color = 'k', horizontalalignment = 'center', verticalalignment = 'center', if_background = False, bg_facecolor = 'white', bg_alpha = 0.2, bg_edgecolor = 'None'):
    t = ax.text(x, y, text, transform = ax.transAxes, color = color, horizontalalignment = horizontalalignment, verticalalignment = verticalalignment)
    if if_background: t.set_bbox(dict(facecolor = bg_facecolor, alpha = bg_alpha, edgecolor = bg_edgecolor))

def add_line(ax, loc, linestyle = '--', color = 'k', alpha = 0.5, direction = 'h', bmin = 0, bmax = 1):
    if direction.lower() in ['h', 'horizontal']:
        ax.axhline(loc, linestyle = linestyle, color = color, alpha = alpha, xmin = bmin, xmax = bmax)
    elif direction.lower() in ['v', 'vertical']:
        ax.axvline(loc, linestyle = linestyle, color = color, alpha = alpha, ymin = bmin, ymax = bmax)
    else:
        raise Exception('Options: h, horizontal, v, and vertical!')

def shift_axis_label(ax, which, x_shift, y_shift):
    if which == 'x':
        ax.xaxis.set_label_coords(x_shift, y_shift, transform = ax.transAxes)
    elif which == 'y':
        ax.yaxis.set_label_coords(x_shift, y_shift, transform = ax.transAxes)
    else:
        raise Exception('wrong axis')

def init_sci_env(fontsize = 14, linemarkersize = 2, legendtitle_fontsize = 14, figuresize = (10, 6), pandas_max_columns = None):
    # plt.rcParams["figure.figsize"] = (10, 6)
    pd.set_option('display.max_columns', pandas_max_columns)
    plt.rcParams.update({
        "font.size": fontsize, 
        "lines.markersize": linemarkersize, 
        'legend.title_fontsize': legendtitle_fontsize, 
        "figure.figsize": figuresize
    })
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return colors

def reset_sci_env():
    pd.reset_option('all')
    plt.rcParams.update(plt.rcParamsDefault)

def savefig(fig, savefile, dpi = 600, bbox_inches = "tight", transparent = False, **kwargs):
    fig.savefig(savefile, dpi = dpi, bbox_inches = bbox_inches, transparent = transparent, **kwargs)


from scipy.stats import gaussian_kde

def kde_scatter(ax, dfp, x_name, y_name, frac = 0.3, v_scale = 0.1, vmin = None, vmax = None, cmap = 'RdYlBu_r'):
    dfp = dfp[[x_name, y_name]].dropna().sample(frac = frac).reset_index(drop = True)
    x = dfp[x_name]
    y = dfp[y_name]

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # fig, ax = plt.subplots(1, 1, figsize = (9, 9))
    ax.scatter(x, y, c=z, s=50, vmin = vmin, vmax = vmax, cmap = cmap)

    xl = np.linspace(np.floor(x.min()), np.ceil(x.max()), 1000)
    ax.plot(xl, xl, ls = '-.', color = 'k')

    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    v_min = np.min([x_min, y_min])
    v_max = np.max([x_max, y_max])
    v_ran = v_max - v_min

    ax.set_xlim(v_min - v_ran * v_scale, v_max +  v_ran * v_scale)
    ax.set_ylim(v_min - v_ran * v_scale, v_max +  v_ran * v_scale)

def line_dot(df, ax, colors = None, legend = True, markercolor = 'white', edgecolor = 'k', markersize = 20, zorder = 10):
    if not colors: colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    df.plot(ax = ax, style = '--', color = colors, legend = legend)
    for c in df.columns:
        ax.scatter(df.index, df[c], color = markercolor, edgecolor = edgecolor, s = markersize, zorder = zorder)
    return ax

def custom_cmap(clist, cname = 'custom_cmap', N = 256):
    import matplotlib.colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cname, clist, N=N)
    return cmap

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
nature_colors = ['#E64B35B2', '#4DBBD5B2', '#00A087B2', '#3C5488B2', '#F39B7FB2', '#8491B4B2', '#91D1C2B2', '#DC0000B2', '#7E6148B2', '#B09C85B2'] # alpha = 0.7
nature_colors_01 = ['#E64B3519', '#4DBBD519', '#00A08719', '#3C548819', '#F39B7F19', '#8491B419', '#91D1C219', '#DC000019', '#7E614819', '#B09C8519'] # alpha = 0.1
nature_colors_03 = ['#E64B354C', '#4DBBD54C', '#00A0874C', '#3C54884C', '#F39B7F4C', '#8491B44C', '#91D1C24C', '#DC00004C', '#7E61484C', '#B09C854C'] # alpha = 0.3
nature_colors_05 = ['#E64B357F', '#4DBBD57F', '#00A0877F', '#3C54887F', '#F39B7F7F', '#8491B47F', '#91D1C27F', '#DC00007F', '#7E61487F', '#B09C857F'] # alpha = 0.5
nature_colors_07 = ['#E64B35B2', '#4DBBD5B2', '#00A087B2', '#3C5488B2', '#F39B7FB2', '#8491B4B2', '#91D1C2B2', '#DC0000B2', '#7E6148B2', '#B09C85B2'] # alpha = 0.7
nature_colors_09 = ['#E64B35E5', '#4DBBD5E5', '#00A087E5', '#3C5488E5', '#F39B7FE5', '#8491B4E5', '#91D1C2E5', '#DC0000E5', '#7E6148E5', '#B09C85E5'] # alpha = 0.9
nature_colors_10 = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF', '#8491B4FF', '#91D1C2FF', '#DC0000FF', '#7E6148FF', '#B09C85FF'] # alpha = 1.0

nature_colors_base = {
    'Cinnabar': '#E64B35', 
    'Shakespeare': '#4DBBD5', 
    'PersianGreen': '#00A087',
    'Chambray': '#3C5488',
    'Apricot': '#F39B7F',
    'WildBlueYonder': '#8491B4',
    'MonteCarlo': '#91D1C2',
    'Monza': '#DC0000',
    'RomanCoffee': '#7E6148',
    'Sandrift': '#B09C85',
}


def show_colors(colors, width = 1, height = 1, hspace = 0.05, wspace = 0.05, fontsize = 10):
    import matplotlib.patches
    import matplotlib.collections

    ncolors = len(colors)
    nrow = ncol = np.floor(np.sqrt(ncolors))
    nrow = int(nrow); ncol = int(ncol)
    while nrow * ncol < ncolors:
        nrow = nrow + 1

    xx = np.arange(0, ncol, (width + wspace))
    yy = np.arange(0, nrow, (height + hspace))

    fig, ax = setup_canvas(1, 1, figsize = (6, 6))

    patches = []
    for i, xi in enumerate(xx):
        for j, yi in enumerate(yy):
            cnt = i * nrow + j
            if cnt < ncolors:
                text = colors[cnt]
                color = colors[cnt]
            else:
                text = None
                color = 'None'
            patch = matplotlib.patches.Rectangle((xi, yi), width, height, fill = True, color = color)
            ax.add_patch(patch)
            ax.text(xi + width / 2, yi  + height / 2, text, fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center')

    pc = matplotlib.collections.PatchCollection(patches)
    ax.add_collection(pc)

    ax.relim()
    ax.autoscale_view()

    ax.axis("off")
    return fig, ax
import imp
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

def pprint(values, p = 2):
    try:
        len(values)
        print([np.round(v, p) for v in values])
    except:
        print(np.round(values, p))

def roundit(ipt, precision = 2):
    if isinstance(ipt, list):
        return [i if isinstance(i, str) else np.round(i, precision) for i in ipt]
    elif isinstance(ipt, dict):
        values = [i if isinstance(i, str) else np.round(i, precision) for i in ipt.values()]
        return dict(zip(ipt.keys(), values))
    elif isinstance(ipt, str):
        if ipt.isnumeric():
            return np.round(float(ipt), precision)
        else:
            return ipt
    else:
        try: 
            return np.round(ipt, precision)
        except:
            return ipt

def timeit(cur_iter, whole_counts, begin_time, ratio = 10, time_unit = 'seconds'):
    time_unit_map = {
        'seconds': 's',
        'minutes': 'm',
        'hours': 'h',
        'days': 'd'
    }
    cur_iter += 1
    progress = cur_iter / whole_counts * 100
    if (np.round(progress, 2) % ratio == 0) or (np.round(progress, 2) >= 100):
        now = datetime.now()
        time_dif = now - begin_time
        time_dif = time_dif.total_seconds()
        time_dif = time_dif / (cur_iter / whole_counts) * (100 - progress)
        if time_unit == 'seconds':
            pass
        elif time_unit == 'minutes':
            time_dif = time_dif / 60
        elif time_unit == 'hours':
            time_dif = time_dif / 3600
        elif time_unit == 'days':
            time_dif = time_dif / 3600 / 24
        else:
            raise Exception('time unit must be seconds, minutes, hours, or days!')
        
        time_dif = np.round(time_dif, 2)
        progress = np.round(progress, 2)
        print(f'{progress} %, {time_dif} {time_unit_map[time_unit]}', end = '|')
    if cur_iter >= whole_counts:
        print('', end = '\n')

def stats_summary(df):
    min_ = df.min().to_frame().T
    Q1 = df.quantile(0.25).to_frame().T
    median_ = df.quantile(0.5).to_frame().T
    mean_ = df.mean().to_frame().T
    Q3 = df.quantile(0.75).to_frame().T
    max_ = df.max().to_frame().T
    df_stats = pd.concat([min_, Q1, median_, mean_, Q3, max_])
    df_stats.index = ["Min", "Q1", "Median", "Mean", "Q3", "Max"]
    return df_stats

def stats_measures(x, y, return_dict = False):
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
    mse = mean_squared_error(x, y)
    r2 = rvalue ** 2
    rmse = np.sqrt(mse)
    mbe = (y - x).mean()
    if return_dict:
        return {
            "R2": r2,
            "SLOPE": slope,
            "RMSE": rmse,
            "MBE": mbe
        }
    else:
        return [r2, slope, rmse, mbe]

def stats_measures_full(x, y):
    from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
    # from sklearn.metrics import mean_absolute_percentage_error
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
    mse = mean_squared_error(x, y)
    r2 = rvalue ** 2
    rmse = np.sqrt(mse)
    mbe = (y - x).mean()
    # ----------------------------------------------------------------
    pearsonr = stats.pearsonr(x, y)
    evs = explained_variance_score(x, y)
    me = max_error(x, y)
    mae = mean_absolute_error(x, y)
    msle = mean_squared_log_error(x, y)
    meae = median_absolute_error(x, y)
    r2_score = r2_score(x, y)
    mpd = mean_poisson_deviance(x, y)
    mgd = mean_gamma_deviance(x, y)
    mtd = mean_tweedie_deviance(x, y)
    return {
        "R2": r2,
        "SLOPE": slope,
        "RMSE": rmse,
        "MBE": mbe,
        "INTERCEPT": intercept,
        "PVALUE": pvalue,
        "STDERR": stderr,
        "PEARSON": pearsonr,
        "EXPLAINED_VARIANCE": evs,
        "MAXERR": me,
        "MAE": mae,
        "MSLE": msle,
        "MEDIAN_AE": meae,
        "R2_SCORE": r2_score,
        "MPD": mpd,
        "MGD": mgd,
        "MTD": mtd
    }

def stats_measures_df(df, name1, name2, return_dict = False):
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(df[name1], df[name2])
    mse = mean_squared_error(df[name1], df[name2])
    r2 = rvalue ** 2
    rmse = np.sqrt(mse)
    mbe = (df[name2] - df[name1]).mean()
    if return_dict:
        return {
            "R2": r2,
            "SLOPE": slope,
            "RMSE": rmse,
            "MBE": mbe
        }
    else:
        return [r2, slope, rmse, mbe]

def load_csv(p, fmt = 'yearfirst', index_col = 0, strip_cols = True, duplicated_time = True, missing = -9999., columns = None, **kwargs):
    df = pd.read_csv(p, index_col = index_col, **kwargs)
    if strip_cols:
        df.columns = [c.strip() for c in df.columns]
    if fmt: 
        if fmt == 'dayfirst': fmt = '%d/%m/%y %H:%M:%S'
        if fmt == 'yearfirst': fmt = '%Y-%m-%d %H:%M:%S'
        df.index = pd.to_datetime(df.index, format = fmt)
    if duplicated_time:
        df = df[~df.index.duplicated(keep='first')]
    if missing:
        df = df.replace(missing, np.nan)
    if columns:
        df = df[columns]
    return df

def load_pickle(p):
    with open(p, "rb") as f:
        ds = pickle.load(f)
    return ds

def dump_pickle(ds, p, large = False):
    with open(p, "wb") as f:
        if large:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(ds, f)

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

def upper_legend(ax, xloc = 0.5, yloc = 1.1, ncols = None):
    # Python > 3.7
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if not ncols: ncols = len(labels)
    ax.legend(by_label.values(), by_label.keys(), loc = "upper center", framealpha = 0.1, frameon = True, bbox_to_anchor=(xloc, yloc), ncol = ncols)
    return ax

def rotate_ticks(ax, which, degree):
    ax.tick_params(axis=which, rotation=degree)

def sort_list_by(lista, listb):
    # sort lista by listb
    lista = [i for _, i in sorted(zip(listb, lista))]
    return lista

def add_text(ax, x, y, text, horizontalalignment = 'center', verticalalignment = 'center'):
    ax.text(x, y, text, transform = ax.transAxes, horizontalalignment = horizontalalignment, verticalalignment = verticalalignment)

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

def str2date(str_, format):
    return datetime.strptime(str_, format)

def date2str(date, format):
    return date.strftime(format)

def timedif(dt1, dt2, mode = 'None'):
    dif = dt2 - dt1
    if mode.lower in ['s', 'sec', 'second']:
        return dif.total_seconds()
    elif mode.lower in ['min', 'minute']:
        return dif.total_seconds() / 60
    elif mode.lower in ['h', 'hour']:
        return dif.total_seconds() / 60 / 60
    elif mode.lower in ['d', 'day']:
        return dif.total_seconds() / 60 / 60 / 24
    elif mode.lower in ['mon', 'month']:
        return dif.total_seconds() / 60 / 60 / 24 / 30
    elif mode.lower in ['y', 'yr', 'year']:
        return dif.total_seconds() / 60 / 60 / 24 / 365
    else:
        return dif

def create_folder(des):
    # exist_ok = False: DO NOT make if the directory exists!
    try:
        des.mkdir(mode = 0o777, parents = True, exist_ok = False)
        print(f'Directory made: {des} ')
    except FileExistsError as e:
        print('Not creating, target directory exists!')
        # FileExistsError

def format_axis_datetime(ax, fmt = '%m/%Y', which = 'x'):
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter(fmt)
    if which == 'x':
        ax.xaxis.set_major_formatter(myFmt)
    else:
        ax.yaxis.set_major_formatter(myFmt)

def nrow_x_ncols(acnt):
    # auto nrows and ncols
    nc = int(np.ceil(np.sqrt(acnt)))
    if nc*(nc-1) >= acnt:
        nr = nc - 1
    else:
        nr = nc
    return nr, nc

def drop_duplicated_row_cols(df, axis):
    df = df.copy()
    if axis == 0:
        # drop rows with duplicated index
        df = df[~df.index.duplicated(keep='first')]
    else:
        df = df.loc[:,~df.columns.duplicated()]
    return df

def intersect_lists(d, method = 1):
    if method == 1:
        # Intersection of multiple lists
        # d = [list1, list2, ...]
        # solution1:
        d = list(set.intersection(*map(set,d)))
    else:
        # solution2:
        from functools import reduce
        # apply intersect1d to (a list of) multiple lists:
        d = list(reduce(np.intersect1d, d))
    return d

def drop_all_vals(df, val = 0, axis = 1):
    # default: drop rows all equal zero
    df = df.copy()
    return df.loc[~(df==val).all(axis=axis)]


def df_replace_dict(df, column, dict_):
    # Replace columns values by dict
    # example: dict_ = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'}
    # df["seasons"] = df["seasons"].map(dict_)
    df = df.copy()
    df[column] = df[column].map(dict_)
    return df

def get_handles_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    return handles, labels

def reorder_labels(handles, labels, ncol):
    leng = len(labels)
    nrow = np.ceil(len(labels) / ncol).astype(int)
    labels_new = labels + [0 for i in range(nrow * ncol - leng)]
    labels_new[0: len(labels)] = labels
    labels_new = np.array(labels_new).reshape(nrow, ncol).T.ravel()
    idx = np.where(labels_new != '0')
    labels_new = labels_new[idx]
    
    handles_new = handles + [0 for i in range(nrow * ncol - leng)]
    handles_new[0: len(handles)] = handles
    handles_new = np.array(handles_new).reshape(nrow, ncol).T.ravel()
    
    idx = np.where(handles_new != 0)
    handles_new = handles_new[idx]
    
    # labels_new = np.resize(np.array(labels), (nrow, ncol)).T
    # handles_new = np.resize(np.array(handles), (nrow, ncol)).T
    # labels_new = np.unique(labels_new.ravel())
    # # handles_new = np.unique(handles_new.ravel())
    
    return handles_new, labels_new

def get_quantile_index(s, q):
    # OR: s[s == s.quantile(.5, interpolation='lower')]
    return (s.sort_values()[::-1] <= s.quantile(.5)).idxmax()

def unify_xylim(ax):
    xylim = np.vstack([ax.get_xlim(), ax.get_ylim()])
    vmin = xylim[:, 0].min()
    vmax = xylim[:, 1].max()
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

def df_sort_user_order(df, order, columns, user_col):
    df = df.copy()
    target_idx = columns.index(user_col)
    columns[target_idx] = 'temp'
    order = dict(zip(order, np.arange(len(order))))
    df['temp'] = df[user_col].map(order)
    df = df.sort_values(by = columns)
    df = df.drop('temp', axis = 1)
    return df
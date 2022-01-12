import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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

def load_csv(p, fmt = 'yearfirst', index_col = 0, strip_cols = True, duplicated_time = 'True'):
    df = pd.read_csv(p, index_col = index_col)
    if strip_cols:
        df.columns = [c.strip() for c in df.columns]
    if fmt: 
        if fmt == 'dayfirst': fmt = '%d/%m/%y %H:%M:%S'
        if fmt == 'yearfirst': fmt = '%Y-%m-%d %H:%M:%S'
        df.index = pd.to_datetime(df.index, format = fmt)
    if duplicated_time:
        df = df[~df.index.duplicated(keep='first')]
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

def setup_canvas(nx, ny, figsize = (10, 6), sharex = True, sharey = True, markersize = 2, fontsize = 16, wspace = 0, hspace = 0, panels = False):
    plt.rcParams.update({'lines.markersize': markersize, 'font.size': fontsize})
    fig, axes = plt.subplots(nx, ny, figsize = figsize, sharex = sharex, sharey = sharey)
    if nx * ny > 1: axes = axes.flatten()

    plt.subplots_adjust(wspace = wspace, hspace = hspace)

    if (nx * ny > 1) & panels:
        for i in range(len(axes)):
            axes[i].text(0.05, 0.8, f'({chr(97 + i)})', transform = axes[i].transAxes)

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

def upper_legend(ax, yloc = 1.1, ncols = None):
    # Python > 3.7
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if not ncols: ncols = len(labels)
    ax.legend(by_label.values(), by_label.keys(), loc = "upper center", framealpha = 0.1, frameon = True , bbox_to_anchor=(0.5, yloc), ncol = ncols)
    return ax

def rotate_ticks(ax, which, degree):
    ax.tick_params(axis=which, rotation=degree)

def sort_list_by(lista, listb):
    lista = [i for _, i in sorted(zip(listb, lista))]
    return lista
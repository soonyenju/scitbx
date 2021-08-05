import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

def pprint(values, p = 2):
    try:
        len(values)
        print([np.round(v, p) for v in values])
    except:
        print(np.round(values, p))

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
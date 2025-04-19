# import imp
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

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

def quiet():
    import warnings
    warnings.simplefilter('ignore')

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
    
def dateparse(dt, format = '%Y-%m-%d'):
    return pd.to_datetime(dt, format = format)

def get_pathtable(paths, dateformat = '', splitformat = '', indexloc = None, dateloc = None):
    df_path = []
    for p in paths:
        if splitformat != '':
            names = p.stem.split(splitformat)
        else:
            names = p.stem
        if type(names) != list: names = [names]
        df_path.append(names + [p])
    colnames = [f'COL_{i}' for i in range(len(names))] + ['PATH']
    if dateloc != None:
        colnames[dateloc] = 'DATETIME'
    df_path = pd.DataFrame(df_path, columns = colnames)
    if dateformat != '':
        df_path.iloc[:, dateloc] = pd.to_datetime(df_path.iloc[:, dateloc], format = dateformat)
    if indexloc != None:
        df_path = df_path.set_index(colnames[indexloc]).sort_index()
    return df_path

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

def get_quantile_index(s, q):
    # OR: s[s == s.quantile(.5, interpolation='lower')]
    return (s.sort_values()[::-1] <= s.quantile(.5)).idxmax()

def df_sort_user_order(df, order, columns, user_col):
    df = df.copy()
    target_idx = columns.index(user_col)
    columns[target_idx] = 'temp'
    order = dict(zip(order, np.arange(len(order))))
    df['temp'] = df[user_col].map(order)
    df = df.sort_values(by = columns)
    df = df.drop('temp', axis = 1)
    return df

from scipy.optimize import curve_fit

def get_curve_fit_r2(func, parameters, x, y):
    from sklearn.metrics import r2_score
    y_pred = func(x, *parameters)
    return r2_score(y, y_pred)


def get_curve_fit_p_value(func, parameters, x, y):
    import scipy.odr
    import scipy.stats
    def f_wrapper_for_odr(beta, x): # parameter order for odr
        return func(x, *beta)

    model = scipy.odr.odrpack.Model(f_wrapper_for_odr)
    data = scipy.odr.odrpack.Data(x,y)
    myodr = scipy.odr.odrpack.ODR(data, model, beta0=parameters,  maxit=0)
    myodr.set_job(fit_type=2)
    parameterStatistics = myodr.run()
    df_e = len(x) - len(parameters) # degrees of freedom, error
    cov_beta = parameterStatistics.cov_beta # parameter covariance matrix from ODR
    sd_beta = parameterStatistics.sd_beta * parameterStatistics.sd_beta
    ci = []
    t_df = scipy.stats.t.ppf(0.975, df_e)
    ci = []
    for i in range(len(parameters)):
        ci.append([parameters[i] - t_df * parameterStatistics.sd_beta[i], parameters[i] + t_df * parameterStatistics.sd_beta[i]])

    tstat_beta = parameters / parameterStatistics.sd_beta # coeff t-statistics
    pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values

    # for i in range(len(parameters)):
    #     print('parameter:', parameters[i])
    #     print('   conf interval:', ci[i][0], ci[i][1])
    #     print('   tstat:', tstat_beta[i])
    #     print('   pstat:', pstat_beta[i])
    #     print()
    df_fit_p = []
    for i in range(len(parameters)):
        df_fit_p.append([parameters[i], ci[i][0], ci[i][1], tstat_beta[i], pstat_beta[i]])
    return pd.DataFrame(df_fit_p, columns = ['parameter', 'conf min', 'conf max', 'tstat', 'pstat'])

def get_github_file(url, target_directory):
    import requests, zipfile, io
    '''
    Get zipped file from github and extract to target directory, while delete the zip file.
    The URL needs to point to the raw file
    Example url: url = "https://github.com/soonyenju/scitbx/raw/refs/heads/master/scitbx/data/Ameriflux_meta.zip"
    '''
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        # Extract ZIP contents
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(target_directory)  # Extract to a folder
        return 'success'

    except Exception as e:
        return e

def unzip(zip_path, extract_path):
    import zipfile
    # Open and extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
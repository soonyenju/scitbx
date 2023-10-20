'''
Basic scientific computing import
'''
import os, sys, subprocess, shutil, warnings, pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from datetime import datetime, timedelta

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import rasterio as rio
except Exception as e:
    print(e)
    install('rasterio')
    import rasterio as rio

def option_load(opt = 0):
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    def geo_load():
        global rio, gpd, xr, rxr
        try:
            import rasterio as rio
        except Exception as e:
            print(e)
            install('rasterio')
            import rasterio as rio

        try:
            import geopandas as gpd
        except Exception as e:
            print(e)
            install('geopandas')
            import geopandas as gpd
        try:
            import xarray as xr
        except Exception as e:
            print(e)
            install('xarray')
            import xarray as xr
        try:
            import rioxarray as xr
        except Exception as e:
            print(e)
            install('rioxarray')
            import rioxarray as rxr

    def plot_load():
        global sns
        try:
            import seaborn as sns
        except Exception as e:
            print(e)
            install('seaborn')
            import seaborn as sns

    def miscellaneous_load():
        global pip, imod
        import pip
        from operator import imod

    if opt == 0:
        pass
    elif opt == 1:
        geo_load()
    elif opt == 2:
        geo_load()
        plot_load()
    elif opt == 3:
        geo_load()
        plot_load()
        miscellaneous_load()
    else:
        raise  Exception(
            '''
            opt should be one of the following: 
            >>> 0: no action (default)
            >>> 1: loading rasterio, geopandas, xarray, and rioxarray
            >>> 2: 1 + seaborn
            >>> 3: 2 + pip and imod
            '''
        )
    print('Selected packages are installed...')


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
nature_colors = ["#E64B35B2", "#4DBBD5B2", "#00A087B2", "#3C5488B2", "#F39B7FB2", "#8491B4B2", "#91D1C2B2", "#DC0000B2", "#7E6148B2"]

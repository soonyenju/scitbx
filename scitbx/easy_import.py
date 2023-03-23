import pip
from operator import imod
import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from scipy import stats
from datetime import datetime, timedelta
try:
    import seaborn as sns
except Exception as e:
    print(e)
    install('seaborn')
    import seaborn as sns
try:
    import xarray as xr
except Exception as e:
    print(e)
    install('xarray')
    import xarray as xr

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
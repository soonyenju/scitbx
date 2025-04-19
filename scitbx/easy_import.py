'''
Basic scientific computing import
'''

# Standard library imports
import gc
import os
import pickle
import pip
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Local application imports
from .commons import *
from .sciplt import *
from .utils import *
from .sciplt import *


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_github(package_url):
    # Install using pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_url])

try:
    import rasterio as rio
except ModuleNotFoundError as e:
    print(f"{e}, so `scitbx` is installing it for you...")
    install('rasterio')
    import rasterio as rio
try:
    import geopandas as gpd
except ModuleNotFoundError as e:
    print(f"{e}, so `scitbx` is installing it for you...")
    install('geopandas')
    import geopandas as gpd
try:
    import xarray as xr
except ModuleNotFoundError as e:
    print(f"{e}, so `scitbx` is installing it for you...")
    install('xarray')
    install('netCDF4')
    import xarray as xr
try:
    import rioxarray as rxr
except ModuleNotFoundError as e:
    print(f"{e}, so `scitbx` is installing it for you...")
    install('rioxarray')
    import rioxarray as rxr
try:
    import seaborn as sns
except ModuleNotFoundError as e:
    print(f"{e}, so `scitbx` is installing it for you...")
    install('seaborn')
    import seaborn as sns

# install('scigeo')
install_github("git+https://github.com/soonyenju/scigeo.git")
from scigeo import *
# install('scieco')
install_github("git+https://github.com/soonyenju/scieco.git")
from scieco import *
# install('sciml')
install_github("git+https://github.com/soonyenju/sciml.git")
from sciml import *
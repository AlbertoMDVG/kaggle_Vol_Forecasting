# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import pyarrow.parquet as pq
import gc
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf

def plot_distribution_with_stats(data, bins=60, title=None):
    min_val = data.min()
    max_val = data.max()
    q25 = data.quantile(0.25)
    median = data.quantile(0.50)
    q75 = data.quantile(0.75)
    
    plt.figure(figsize=(4,4))
    plt.hist(data, bins=bins)
    plt.axvline(min_val, linestyle='--')
    plt.axvline(q25, linestyle='--')
    plt.axvline(median, linestyle='--')
    plt.axvline(q75, linestyle='--')
    plt.axvline(max_val, linestyle='--')
    
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()



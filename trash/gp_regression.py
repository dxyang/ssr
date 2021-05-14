from datetime import datetime, timedelta
import glob
import os
import sys
import time

import cartopy.crs as ccrs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

cwd = os.getcwd()
sys.path.append(f"{cwd}/forecast_rodeo")
sys.path.append(f"{cwd}/forecast_rodeo/src/experiments")
from experiments_util import get_target_date, month_day_subset
from stepwise_util import default_stepwise_candidate_predictors

def get_data_for_regression():
    #https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    '''
    some setup
    '''
    target = "contest_tmp2m" # "contest_precip" or "contest_tmp2m"
    target_horizon = "34w" # "34w" or "56w"

    data_path = os.path.expanduser("forecast_rodeo/results/regression/shared")
    data_matrices_folder = f"{target}_{target_horizon}"
    fs = glob.glob(f"{data_path}/{data_matrices_folder}/*.h5")
    lat_lon_date_data_file = fs[0]
    date_data_file = fs[1]

    '''
    dates of interest and other variables
    '''
    submission_dates = [datetime(y,4,18)+timedelta(14*i) for y in range(2011,2018) for i in range(26)]
    submission_dates = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day) for date in submission_dates]
    target_date_objs = [get_target_date(submission_date_str, target_horizon) for submission_date_str in submission_dates]
    target_dates = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day) for date in target_date_objs]

    submission_date = submission_dates[0]
    target_date_obj = target_date_objs[0]
    target_date = target_dates[0]

    # some vars
    gt_col = target.split('_')[-1]  # 'tmp2m'
    clim_col = f"{gt_col}_clim"     # 'tmp2m_clim'
    anom_col = f"{gt_col}_anom"     # 'tmp2m_anom'
    base_col = 'zeros'
    group_by_cols = ['lat', 'lon']
    first_train_year = 1978 # use 1948 for precip, 1978 for temp
    start_delta = 29 # 29 for 34w or 43 for 56w
    last_train_date = target_date_obj - timedelta(start_delta)
    print(f"submission date: {submission_date}")
    print(f"target date: {target_date}")
    print(f"last train date: {last_train_date}")

    # get data array names we care about
    candidate_x_cols = default_stepwise_candidate_predictors(target, target_horizon, hindcast=False)

    relevant_cols = set(candidate_x_cols
                        +[base_col,clim_col,anom_col,'start_date','lat','lon','target','year','ones']
                        +group_by_cols)

    '''
    load the data
    '''
    # raw data files
    date_data = pd.read_hdf(date_data_file)
    lat_lon_date_data = pd.read_hdf(lat_lon_date_data_file)

    # filter out data older than "first_train_year" and keep only relevant columns
    data = lat_lon_date_data.loc[lat_lon_date_data.start_date.dt.year >= first_train_year,
                                lat_lon_date_data.columns.isin(relevant_cols)]
    data = pd.merge(data, date_data.loc[date_data.start_date.dt.year >= first_train_year,
                                        date_data.columns.isin(relevant_cols)],
                    on="start_date", how="left")
    del lat_lon_date_data
    del date_data

    print(len(data.columns))
    print(data.columns)

    # filter to days within margin around target date
    margin_in_days = 56
    print(f"target date: {target_date_obj}, margin in days: {margin_in_days}")
    sub_data = month_day_subset(data, target_date_obj, margin_in_days).copy()
    del data

    # a bit of munging around
    sub_data['year'] = sub_data.start_date.dt.year
    sub_data['ones'] = 1.0
    sub_data['zeros'] = 0.0

    # this is really tmp2m_clim + tmp2m_anom
    sub_data['target'] = sub_data[clim_col] + sub_data[anom_col]

    # drop data that doesn't have valid targets
    sub_data_valid_targets = sub_data.dropna(subset=candidate_x_cols+['target'])
    print(sub_data_valid_targets.head())

    # grouping
    data_grouped_by_latlon = sub_data_valid_targets.loc[:,relevant_cols].groupby(group_by_cols)
    lat_oi, lon_oi = (37.0, 238.0)

    data_at_lat_lon = data_grouped_by_latlon.get_group((lat_oi, lon_oi))

    # return just X, Y, and dates
    Y = data_at_lat_lon['target']
    X = data_at_lat_lon[candidate_x_cols]
    dates = data_at_lat_lon['start_date']

    return X, Y, dates

def run_gp_regression(X, Y):
    kernel = C(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    test = gp.fit(X.to_numpy(), Y.to_numpy())
    import pdb; pdb.set_trace()

    print(f"done fitting")

if __name__ == "__main__":
    X, Y, dates = get_data_for_regression()
    print("starting to run regression")
    run_gp_regression(X, Y)
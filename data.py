from datetime import datetime, timedelta
import glob
import os
import math
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

cwd = os.getcwd()
sys.path.append(f"{cwd}/forecast_rodeo")
sys.path.append(f"{cwd}/forecast_rodeo/src/experiments")
from stepwise_util import default_stepwise_candidate_predictors


'''
time helpers
'''
def month_to_float(month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31.0
    elif month in [2]:
        return 28.0
    elif month in [4, 6, 9, 11]:
        return 30.0

def dt_to_float(dt):
    year = dt.year
    month = np.sum([month_to_float(mth) for mth in np.arange(1, dt.month)])
    day = dt.day
    val = float(year) + ((month + day) / 365.0)
    return val

def float_to_dt(dt_float):
    year = math.floor(dt_float)
    monthday_remainder = (dt_float - year) * 365.0
    month = 1
    for numdaysinmonth in [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]:
        if monthday_remainder > numdaysinmonth:
            monthday_remainder -= numdaysinmonth
            month += 1
        else:
            break
    day = math.floor(monthday_remainder)

    dt = datetime(year, month, day)
    return dt

def get_prediction_dates(subsample_rate = 1):
    # make 3-4 predictions from these dates!
    submission_dates = [datetime(y,4,18)+timedelta(14*i) for y in range(2011,2018) for i in range(26)]
    submission_dates_strs = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day) for date in submission_dates]
    return submission_dates[::subsample_rate], submission_dates_strs[::subsample_rate]


'''
dataset
'''
def get_data(add_ones=False):
    '''
    setup
    '''
    target = "contest_tmp2m" # "contest_precip" or "contest_tmp2m"
    target_horizon = "34w" # "34w" or "56w"

    data_path = os.path.expanduser("forecast_rodeo/results/regression/shared")
    data_matrices_folder = f"{target}_{target_horizon}"
    fs = glob.glob(f"{data_path}/{data_matrices_folder}/*.h5")
    print(fs)
    lat_lon_date_data_file = fs[0]
    date_data_file = fs[1]

    # some vars
    gt_col = target.split('_')[-1]  # 'tmp2m'
    clim_col = f"{gt_col}_clim"     # 'tmp2m_clim'
    anom_col = f"{gt_col}_anom"     # 'tmp2m_anom'
    group_by_cols = ['lat', 'lon']
    first_train_year = 1978 # use 1948 for precip, 1978 for temp
    start_delta = 29 # 29 for 34w or 43 for 56w

    # get data array names we care about
    candidate_x_cols = default_stepwise_candidate_predictors(target, target_horizon, hindcast=False)

    relevant_cols = set(candidate_x_cols
                        +[clim_col,anom_col,'start_date','lat','lon','target','year']
                        +group_by_cols)
    if not add_ones:
        relevant_cols.remove('ones')
        candidate_x_cols.remove('ones')

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

    data['year'] = data.start_date.dt.year
    data['ones'] = 1.0
    # this is really tmp2m_clim + tmp2m_anom
    data['target'] = data[clim_col] + data[anom_col]

    # drop data that doesn't have valid targets
    data_valid_targets = data.dropna(subset=candidate_x_cols+['target'])

    data_grouped_by_latlon = data_valid_targets.loc[:,relevant_cols].groupby(group_by_cols)

    latlons = [latlon for latlon, _ in data_grouped_by_latlon]
    lat_oi, lon_oi = latlons[0] #27.0 261.0

    data_at_lat_lon = data_grouped_by_latlon.get_group((lat_oi, lon_oi))
    print(f"lat_oi: {lat_oi}, lon_oi: {lon_oi}")

    '''
    split to train and test
    '''
    anoms = data_at_lat_lon[anom_col]
    clims = data_at_lat_lon[clim_col]
    temps = data_at_lat_lon['target']
    X = data_at_lat_lon[candidate_x_cols]
    dates = data_at_lat_lon['start_date']
    dates_np = np.array([
        date.to_pydatetime() for date in dates
    ])

    years_in_data = dates.dt.year
    first_year = min(years_in_data)
    last_year = max(years_in_data)

    Xnp = X.to_numpy().astype(np.float32)
    anoms_np = anoms.to_numpy().astype(np.float32)
    clims_np = clims.to_numpy().astype(np.float32)
    temps_np = temps.to_numpy().astype(np.float32)

    columnstr_to_index = {key: idx for idx, key in enumerate(X.columns)}
    index_to_columnstr = {idx: key for idx, key in enumerate(X.columns)}

    return (
        Xnp, anoms_np, clims_np, temps_np, dates_np, \
        columnstr_to_index, index_to_columnstr \
    )


from datetime import datetime, timedelta
import glob
import os
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

def get_data():
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
    # this is really tmp2m_clim + tmp2m_anom
    data['target'] = data[clim_col] + data[anom_col]

    # drop data that doesn't have valid targets
    data_valid_targets = data.dropna(subset=candidate_x_cols+['target'])

    data_grouped_by_latlon = data_valid_targets.loc[:,relevant_cols].groupby(group_by_cols)

    latlons = [latlon for latlon, _ in data_grouped_by_latlon]
    lat_oi, lon_oi = latlons[0] #(37.0, 238.0)

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
        Xnp, anoms_np, clims_np, temps_np, dates, \
        columnstr_to_index, index_to_columnstr \
    )


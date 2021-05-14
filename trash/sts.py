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

from data import get_data

cwd = os.getcwd()
sys.path.append(f"{cwd}/forecast_rodeo")
sys.path.append(f"{cwd}/forecast_rodeo/src/experiments")

def build_model(observed_time_series):
    seasonal = tfp.sts.Seasonal(
        num_seasons=4,
        observed_time_series=observed_time_series,
        num_steps_per_season=91,
        name='seasonal'
    )
    monthly = tfp.sts.Seasonal(
        num_seasons=12,
        observed_time_series=observed_time_series,
        num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
        drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
        initial_effect_prior=tfd.Normal(loc=0., scale=5.),
        name='month_of_year',
    )
    features_effect = tfp.sts.LinearRegression(
        design_matrix=X_train - np.mean(X_train),
        name='other_features'
    )
    autoregressive = tfp.sts.Autoregressive(
        order=1,
        observed_time_series=observed_time_series,
        name='autoregressive'
    )
    model = tfp.sts.Sum([
            seasonal,
            monthly,
            features_effect,
            autoregressive
        ],
        observed_time_series=observed_time_series
    )
    return model

def run_sts()
    '''
    sts forecasting
    '''
    X, Y, dates, \
    X_train, Y_train, dates_train, \
    X_test, Y_test, dates_test, \
    columnstr_to_index, index_to_columnstr = get_data()

    years_in_data = dates.dt.year
    first_year = min(years_in_data)
    last_year = max(years_in_data)

    temperature_model = build_model(Y_train)

    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=temperature_model)

    # Allow external control of optimization to reduce test runtimes.
    num_variational_steps = 100
    print(num_variational_steps)

    optimizer = tf.optimizers.Adam(learning_rate=.1)

    # Using fit_surrogate_posterior to build and optimize the variational loss function.
    @tf.function(experimental_compile=True)
    def train():
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=temperature_model.joint_log_prob(
            observed_time_series=Y_train),
        surrogate_posterior=variational_posteriors,
        optimizer=optimizer,
        num_steps=num_variational_steps)
    return elbo_loss_curve

    start = time.time()
    elbo_loss_curve = train()
    end = time.time()
    print(f"{end - start} seconds")
    plt.plot(elbo_loss_curve)
    plt.savefig("test.png")
    import pdb; pdb.set_trace()
    # Draw samples from the variational posterior.
    q_samples_demand_ = variational_posteriors.sample(50)

if __name__ == "__main__":
    run_sts()
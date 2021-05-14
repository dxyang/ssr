from datetime import datetime, timedelta
import glob
from itertools import islice
import os
import pickle
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from tqdm import tqdm

from data import get_data, get_prediction_dates, dt_to_float, float_to_dt

'''
Load data and split into training set for a prediction date
'''
X_original, anoms, clims, temps, dates, \
columnstr_to_index, index_to_columnstr = get_data(add_ones=False)
dates_as_float = np.array([dt_to_float(d) for d in dates]).astype(np.float64)

prediction_dates, prediction_dates_strs = get_prediction_dates(subsample_rate=2)

'''
Do 3-4 week temperature predictions
'''
results_dict = {d: {} for d in prediction_dates}
skills = []
for num, prediction_date in enumerate(prediction_dates):
    it_start = time.time()
    print(f"[{num} / {len(prediction_dates)}] ----- {prediction_dates_strs[num]} -----")

    # ~10 years of training data
    useful_idxs = dates >= datetime(prediction_date.year - 10, 1, 1)
    train_idxs = np.logical_and(dates <= prediction_date, useful_idxs)
    test_idxs = np.logical_and(dates > prediction_date, useful_idxs)

    print(useful_idxs)
    print(train_idxs)
    print(test_idxs)
    import pdb; pdb.set_trace()

    anoms_train = anoms[train_idxs]
    clims_train = clims[train_idxs]
    temps_train = temps[train_idxs]
    dates_train = dates[train_idxs]
    dates_as_float_train = dates_as_float[train_idxs]

    anoms_test = anoms[test_idxs][:28]
    clims_test = clims[test_idxs][:28]
    temps_test = temps[test_idxs][:28]
    dates_test = dates[test_idxs][:28]
    dates_as_float_test = dates_as_float[test_idxs][:28]

    print(f"Num in training set: {len(anoms_train)}")
    print(f"Num in testing set: {len(anoms_test)}")

    is_predicting_anomaly = False
    if is_predicting_anomaly:
        Y_train = anoms_train
        Y_test = anoms_test
        print(f"Y is temperature anomaly!")
    else:
        Y_train = temps_train
        Y_test = temps_test
        print(f"Y is temperature directly!")

    normalize_features = True
    X_train = np.copy(X_original[train_idxs])
    X_test = np.copy(X_original[test_idxs][:28])
    if normalize_features:
        for col_idx in range(X_train.shape[1]):
            if index_to_columnstr[col_idx] == 'ones':
                continue
            vals_train = X_train[:, col_idx]
            vals_test = X_test[:, col_idx]
            mean = np.mean(X_train[:, col_idx])
            std = np.std(X_train[:, col_idx])
            X_train[:, col_idx] = (vals_train - mean) / std
            X_test[:, col_idx] = (vals_test - mean) / std
    X = np.concatenate((X_train, X_test), axis=0)

    '''
    Define the STS model
    '''
    def build_model(observed_time_series):
        annual = tfp.sts.SmoothSeasonal(
            period=365,
            frequency_multipliers=[1, 2, 4, 12],
            initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([8])),
            observed_time_series=observed_time_series,
            name='annual'
        )
        features_effects = []
        features_effects.append(
            tfp.sts.LinearRegression(
                design_matrix=X,# - np.mean(X),
                name=f"linear_regression"
            )
        )
        # for feature_name, idx in columnstr_to_index.items():
        #     features_effects.append(
        #         tfp.sts.LinearRegression(
        #             design_matrix=np.expand_dims(X[:, idx], 1),# - np.mean(X),
        #             name=f"lr_{feature_name}_{idx}"
        #         )
        #     )
        sts_components = [
            annual,
        ] + features_effects
        model = tfp.sts.Sum(
            sts_components,
            observed_time_series=observed_time_series
        )
        return model

    temperature_model = build_model(Y_train)

    '''
    VI for component parameters
    '''
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=temperature_model)

    num_variational_steps = 60
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
    print(f"{end - start} seconds for training")

    '''
    Do prediction with trained model
    '''
    # Draw samples from the variational posterior.
    q_samples_temperature_ = variational_posteriors.sample(25)

    num_forecast_steps = 7 * 4 # 28 days

    start_forecast = time.time()
    temperature_forecast_dist = tfp.sts.forecast(
        model=temperature_model,
        observed_time_series=Y_train,
        parameter_samples=q_samples_temperature_,
        num_steps_forecast=num_forecast_steps)

    num_samples=10
    (
        temperature_forecast_mean,
        temperature_forecast_scale,
        temperature_forecast_samples
    ) = (
        temperature_forecast_dist.mean().numpy()[..., 0],
        temperature_forecast_dist.stddev().numpy()[..., 0],
        temperature_forecast_dist.sample(num_samples).numpy()[..., 0]
    )
    end_forecast = time.time()
    print(f"{end_forecast - start_forecast} seconds for forecasting")

    '''
    Calculate the skill
    '''
    anoms_hat = temperature_forecast_mean[14:] - clims_test[14:num_forecast_steps]
    anoms_gt = anoms_test[14:num_forecast_steps]
    def calculate_skill(a_hat, a):
        return a_hat.dot(a) / (np.linalg.norm(a_hat) * np.linalg.norm(a))

    skill = calculate_skill(anoms_hat.squeeze(), anoms_gt.squeeze())

    it_end = time.time()

    print(f"time elapsed: {it_end - it_start} seconds, skill: {skill}")

    '''
    Bookkeeping
    '''
    # Show values of parameters found
    params_dict = {}
    for param in temperature_model.parameters:
        d = {
            "mean": np.mean(q_samples_temperature_[param.name], axis=0),
            "std": np.std(q_samples_temperature_[param.name], axis=0)
        }
        params_dict[param.name] = d

    skills.append(skill)
    result = {
        "skill": skill,
        "forecast_temps": temperature_forecast_mean,
        "forecast_days": dates_test[:num_forecast_steps],
        "params_dict": params_dict
    }
    results_dict[prediction_date] = result

skills = np.array(skills)
temps = temperature_forecast_mean

np.save("results_sts_temperature_34_skills_2017.npy", skills)
np.save("results_sts_temperature_34_temps_2017.npy", temps)

print(f"all skills: {skills}")
print(f"avg skills: {np.mean(skills)}, std: {np.std(skills)}")

with open('results_sts_temperature_34_2017.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle)

import pdb; pdb.set_trace()
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

prediction_dates, prediction_dates_strs = get_prediction_dates(subsample_rate=10)

'''
Do 3-4 week temperature predictions
'''
results_dict = {d: {} for d in prediction_dates}
skills = []
for num, prediction_date in enumerate(prediction_dates):
    start = time.time()
    print(f"[{num} / {len(prediction_dates)}] ----- {prediction_dates_strs[num]} -----")

    train_idxs = dates <= prediction_date
    test_idxs = dates > prediction_date

    X_original_train = X_original[train_idxs, :]
    anoms_train = anoms[train_idxs]
    clims_train = clims[train_idxs]
    temps_train = temps[train_idxs]
    dates_train = dates[train_idxs]
    dates_as_float_train = dates_as_float[train_idxs]

    X_original_test = X_original[test_idxs, :]
    anoms_test = anoms[test_idxs]
    clims_test = clims[test_idxs]
    temps_test = temps[test_idxs]
    dates_test = dates[test_idxs]
    dates_as_float_test = dates_as_float[test_idxs]

    print(f"Num in training set: {len(anoms_train)}")
    print(f"Num in testing set: {len(anoms_test)}")

    is_predicting_anomaly = False
    if is_predicting_anomaly:
        Y_train = anoms_train.astype(np.float64)
        Y_test = anoms_test.astype(np.float64)
        print(f"Y is temperature anomaly!")
    else:
        Y_train = temps_train.astype(np.float64)
        Y_test = temps_test.astype(np.float64)
        print(f"Y is temperature directly!")

    '''
    Define the GP model
    '''
    # Define mean function which is the means of observations
    observations_mean = tf.constant(
        [np.mean(Y_train)], dtype=tf.float64)
    mean_fn = lambda _: observations_mean

    # Define the kernel with trainable parameters.
    # Note we transform some of the trainable variables to ensure
    #  they stay positive.

    # Use float64 because this means that the kernel matrix will have
    #  less numerical issues when computing the Cholesky decomposition

    # Constrain to make sure certain parameters are strictly positive
    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    # Smooth kernel hyperparameters
    smooth_amplitude = tfp.util.TransformedVariable(
        initial_value=10., bijector=constrain_positive, dtype=np.float64,
        name='smooth_amplitude')
    smooth_length_scale = tfp.util.TransformedVariable(
        initial_value=10., bijector=constrain_positive, dtype=np.float64,
        name='smooth_length_scale')

    # Smooth kernel
    smooth_kernel = tfk.ExponentiatedQuadratic(
        amplitude=smooth_amplitude,
        length_scale=smooth_length_scale)

    # Local periodic kernel hyperparameters
    periodic_amplitude = tfp.util.TransformedVariable(
        initial_value=5.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_amplitude')
    periodic_length_scale = tfp.util.TransformedVariable(
        initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_length_scale')
    periodic_period = tfp.util.TransformedVariable(
        initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_period')
    periodic_local_length_scale = tfp.util.TransformedVariable(
        initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_local_length_scale')
    # Local periodic kernel
    local_periodic_kernel = (
        tfk.ExpSinSquared(
            amplitude=periodic_amplitude,
            length_scale=periodic_length_scale,
            period=periodic_period) *
        tfk.ExponentiatedQuadratic(
            length_scale=periodic_local_length_scale))

    # Noise variance of observations
    # Start out with a medium-to high noise
    observation_noise_variance = tfp.util.TransformedVariable(
        initial_value=1, bijector=constrain_positive, dtype=np.float64,
        name='observation_noise_variance')

    trainable_variables = [v.variables[0] for v in [
        smooth_amplitude,
        smooth_length_scale,
        periodic_amplitude,
        periodic_length_scale,
        periodic_period,
        periodic_local_length_scale,
        observation_noise_variance
    ]]

    # Sum all kernels to single kernel containing all characteristics
    kernel = (smooth_kernel + local_periodic_kernel)

    '''
    Fit the GP
    '''
    # Define mini-batch data iterator
    batch_size = 128

    batched_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (dates_as_float_train.reshape(-1, 1), Y_train))
        .shuffle(buffer_size=len(Y_train))
        .repeat(count=None)
        .batch(batch_size)
    )

    @tf.function(autograph=False, experimental_compile=False)  # Use tf.function for more effecient function evaluation
    def gp_loss_fn(index_points, observations):
        """Gaussian process negative-log-likelihood loss function."""
        gp = tfd.GaussianProcess(
            mean_fn=mean_fn,
            kernel=kernel,
            index_points=index_points,
            observation_noise_variance=observation_noise_variance
        )

        negative_log_likelihood = -gp.log_prob(observations)
        return negative_log_likelihood

    # Fit hyperparameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    # Training loop
    batch_nlls = []  # Batch NLL for plotting
    full_ll = []  # Full data NLL for plotting
    nb_iterations = 5001
    for i, (index_points_batch, observations_batch) in tqdm(enumerate(islice(batched_dataset, nb_iterations)), file=sys.stdout):
        # Run optimization for single batch
        with tf.GradientTape() as tape:
            loss = gp_loss_fn(index_points_batch, observations_batch)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        batch_nlls.append((i, loss.numpy()))
        # Evaluate on all observations
        if i % 100 == 0:
            # Evaluate on all observed data
            ll = gp_loss_fn(
                index_points=dates_as_float_train.reshape(-1, 1),
                observations=Y_train)
            full_ll.append((i, ll.numpy()))

    '''
    Do prediction with trained model
    '''
    num_forecast_days = 28
    one_day = timedelta(days=1)

    extra_dates = np.array([prediction_date + i * one_day for i in np.arange(1, 29)])
    extra_dates_as_floats = np.array([dt_to_float(d) for d in extra_dates]).reshape(-1, 1)

    prediction_dates_for_gp = dates_as_float_train.reshape(-1, 1)
    prediction_dates_extra = np.concatenate((prediction_dates_for_gp, extra_dates_as_floats))

    # Posterior GP using fitted kernel and observed data
    gp_posterior_predict = tfd.GaussianProcessRegressionModel(
        mean_fn=mean_fn,
        kernel=kernel,
        index_points=extra_dates_as_floats,
        observation_index_points=dates_as_float_train.reshape(-1, 1),
        observations=Y_train,
        observation_noise_variance=observation_noise_variance)

    # Posterior mean and standard deviation
    posterior_mean_predict = gp_posterior_predict.mean()
    posterior_std_predict = gp_posterior_predict.stddev()

    '''
    Calculate the skill
    '''
    anoms_hat = posterior_mean_predict.numpy()[14:] - clims[14:num_forecast_days]
    anoms_gt = anoms_test[14:num_forecast_days]

    def calculate_skill(a_hat, a):
        return a_hat.dot(a) / (np.linalg.norm(a_hat) * np.linalg.norm(a))

    skill = calculate_skill(anoms_hat.squeeze(), anoms_gt.squeeze())

    end = time.time()

    print(f"time elapsed: {end - start} seconds, skill: {skill}")

    '''
    Bookkeeping
    '''
    # Show values of parameters found
    variables = [
        smooth_amplitude,
        smooth_length_scale,
        periodic_amplitude,
        periodic_length_scale,
        periodic_period,
        periodic_local_length_scale,
        observation_noise_variance
    ]
    data = list([(var.variables[0].name[:-2], var.numpy()) for var in variables])

    skills.append(skill)
    result = {
        "skill": skill,
        "forecast_temps": posterior_mean_predict.numpy(),
        "forecast_days": dates_test[:num_forecast_days],
        "kernel_params": data
    }
    results_dict[prediction_date] = result

results_np = np.zeros((2, len(skills)))
skills = np.array(skills)
temps = posterior_mean_predict.numpy()

results_np[0] = skills
results_np[1] = temps

np.save("results_temperature_34_skills.npy", results_np)

print(f"all skills: {skills}")
print(f"avg skills: {np.mean(skills)}, std: {np.std(skills)}")


with open('results_temperature_34.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle)

with open('results_temperature_34.pickle', 'rb') as handle:
    b = pickle.load(handle)

import pdb; pdb.set_trace()
print(results_dict == b)

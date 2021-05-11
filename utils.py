import collections

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


'''
plotting adapted from
https://www.tensorflow.org/probability/examples/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand
'''
def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None, plot_num=None):
    """Plot a forecast distribution against the 'true' time series."""
    colors = sns.color_palette()
    c1, c2, c3 = colors[0], colors[1], colors[2]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    if plot_num is not None:
        x = x[-plot_num:]
        y = y[-plot_num:]

    num_steps = len(y)
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast

    ax.plot(x, y, lw=2, color=c1, label='ground truth')

    forecast_steps = x[num_steps_train:]

    ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c3, alpha=0.5)

    ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
           label='forecast')
    ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

    ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
    yrange = ymax-ymin
    ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
    ax.set_title("{}".format(title))
    ax.legend()

    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()

    return fig, ax

def plot_components(dates,
                component_means_dict,
                component_stddevs_dict,
                x_locator=None,
                x_formatter=None,
                vals_to_plot=None):
  """Plot the contributions of posterior components in a single figure."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  if vals_to_plot is not None:
      dates = dates[-vals_to_plot:]
      for k, v in component_means_dict.items():
          component_means_dict[k] = v[-vals_to_plot:]
      for k, v in component_stddevs_dict.items():
          component_stddevs_dict[k] = v[-vals_to_plot:]


  axes_dict = collections.OrderedDict()
  num_components = len(component_means_dict)
  fig = plt.figure(figsize=(12, 2.5 * num_components))
  for i, component_name in enumerate(component_means_dict.keys()):
    component_mean = component_means_dict[component_name]
    component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
      ax.xaxis.set_major_locator(x_locator)
      ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
  fig.autofmt_xdate()
  fig.tight_layout()
  return fig, axes_dict


def build_model(observed_time_series):
    seasonal = tfp.sts.Seasonal( \
        num_seasons=4, \
        observed_time_series=observed_time_series,\
        num_steps_per_season=91, \
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
#     features_effect = tfp.sts.LinearRegression(
#         design_matrix=other_features - np.mean(other_features),
#         name='other_fehthhhhhhhhhhhatures'
#     )
#     autoregressive = tfp.sts.Autoregressive(
#         order=1,
#         observed_time_series=observed_time_series,
#         name='autoregressive'
#     )
    model = tfp.sts.Sum([
            seasonal,
            monthly,
#             features_effect,
#             autoregressive
        ],
        observed_time_series=observed_time_series
    )
    return model

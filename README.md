# Bayesian Methods for Subseasonal Temperature Forecasting
6.435 Final Project in Spring 2021 by Anubhav Guha and Daniel Yang

We focus on the use of Bayesian models and methods for subseasonal temperature forecasting in the western United States. Subseasonal forecasts, which refer to predictions 2-6 weeks ahead, of weather, temperature and other climate-related quantities are of importance to a variety of stakeholders - from private companies in the agricultural sector to governing bodies hoping to more effectively plan the allocation of water resources. In comparison to short and medium term forecasts (1-14 days) and long term forecasts (> 6 weeks), there exists a relative paucity of effective methods for subseasonal forecasting. 

# Related Work
* Hwang, J., Orenstein, P., Cohen, J., Pfeiffer, K., & Mackey, L. (2019, July). Improving subseasonal forecasting in the western US with machine learning. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2325-2335). [arxiv](https://arxiv.org/abs/1809.07394)

# Repo Setup
# PyEnv + Poetry
PyEnv is a great way to manage local python installations separate from system installations. Poetry will be our dependency manager that creates an environment using a base pyenv python installation.

```
# pyenv
curl https://pyenv.run | bash

# the installer should automatically add these lines to your bashrc or zshrc, restart your shell as needed
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# install needed python version through pyenv
pyenv install 3.8.6
pyenv global 3.8.6 # will make all directories use python 3.8.6

# poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# the installer should automatically add this line to your bashrc or zshrc, restart your shell as needed
export PATH = "$HOME/.poetry/bin:$PATH"
```

To install the dependencies of this repo, simply run `poetry install` from the root of the repo. Run `poetry shell` to active the virtual environment.

# Processed Data
[Fork](https://github.com/dxyang/forecast_rodeo) of the original forecast rodeo repo, with code fixes to work with the Python3 environment in this repo. This was used to generate the original data matrices as used by the paper. The datafiles can be found on Google Drive [here](https://drive.google.com/drive/folders/1bFJ64Q_NlbKc1J1um-tQFmT1xEBBYWhV?usp=sharing) or at `/data/vision/fisher/code/dxyang/ssr/forecast_rodeo/results/regression/shared`.

You can checkout the fork by running
```
git submodule init
git submodule update
```

You should put the original data matrices from Google Drive in the forked repo such that these paths are valid for example:
```
{SSR_REPO_ROOT}/forecast_rodeo/results/regression/shared/contest_tmp2m_34w/lat_lon_date_data-contest_tmp2m_34w.h5
{SSR_REPO_ROOT}/forecast_rodeo/results/regression/shared/contest_tmp2m_34w/date_data-contest_tmp2m_34w.h5
```

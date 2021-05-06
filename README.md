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
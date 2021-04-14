#!/bin/bash
# Script that moves to the project directory and exports all of
# the variables stored in the .env file to the local environment.
#
# Example use:
#    $ ./open_mindpy.sh
#

# Path to project from wherever this will reside
cd /data/vision/fisher/code/dxyang/ssr
echo In directory $(pwd)

# Export the variables in the .env file
echo Exporting environment variables from .env
set -o allexport
[[ -f .env ]] && source .env
set +o allexport

# Activate a new shell with the environment activated
poetry shell

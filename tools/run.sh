#!/bin/bash

# runner of training at GPU server
# usage: ./run.sh
#
# this script will download pipx.pyz and run postprocess.py and train.py by pipx
# see also: postprocess.py, train.py

set -euo pipefail

if [ ! -f pipx.pyz ]; then
    wget https://github.com/pypa/pipx/releases/download/1.7.1/pipx.pyz
fi
python3 pipx.pyz run ./postprocess.py
python3 pipx.pyz run ./train.py

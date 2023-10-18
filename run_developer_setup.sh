#!/bin/bash

export TMPDIR=/home/$USER/tmp
mkdir $TMPDIR
export PIP_CACHE_DIR=/home/$USER/tmp

./clean.sh
python setup.py clean


pip install --cache-dir $TMPDIR -r requirements.txt

python setup.py develop

pip install -e .

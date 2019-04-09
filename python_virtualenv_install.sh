#!/bin/bash

PYTHON_PATH=/usr/bin/python2.7
#PYTHON_PATH=/usr/bin/python3.6

PYTHON_ENV_DIR=$HOME/python_envs

PYTHON_ENV_NAME=c-rnn-env

if [ -z "$PYTHON_ENV_NAME" ]; then
    echo "ERROR: Please define the virtual environment name (PYTHON_ENV_NAME)"
    exit 1
fi

mkdir -p $PYTHON_ENV_DIR

cd $PYTHON_ENV_DIR

virtualenv -p $PYTHON_PATH $PYTHON_ENV_NAME

source $PYTHON_ENV_NAME/bin/activate

pip install gprof2dot

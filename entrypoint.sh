#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-baro}
LOGURU_COLORIZE=0 .venv/bin/python main.py container run

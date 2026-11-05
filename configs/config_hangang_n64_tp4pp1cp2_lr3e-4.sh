#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_gh200.sh

# hyperparameters
export MINIBS=1
export TP=4
export PP=1
export CP=2

# system parameters
export DGXNNODES=64

# convergence within the RCP limits for GBS 32
export LR=0.0003
export MAX_STEPS=1024

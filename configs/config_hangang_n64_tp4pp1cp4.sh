#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_gh200.sh

# hyperparameters
export MINIBS=1
export TP=4
export PP=1
export CP=4
export CP_COMM_TYPE=p2p   # a2a requires KV heads per TP rank (2) divisible by CP

# system parameters
export DGXNNODES=64

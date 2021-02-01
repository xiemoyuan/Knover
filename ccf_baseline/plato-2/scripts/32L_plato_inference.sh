#!/bin/bash
set -eux

export CUDA_VISIBLE_DEVICES=0,1

# change to Knover working directory
SCRIPT=`realpath "$0"`
KNOVER_DIR=`dirname ${SCRIPT}`/../..
cd $KNOVER_DIR

./scripts/distributed/infer.sh ./package/dialog_en/plato/32L_infer.conf

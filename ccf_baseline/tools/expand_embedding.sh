#!/bin/bash
################################################################################
# Expand embedding: `embedding_name` from [old_size, hidden_size] -> [new_size, hidden_size]
# Retain the first `old_size` params.
################################################################################

PYTHONPATH=.

python -u \
    ./tools/expand_embedding.py \
    --param_path "" \
    --save_path ./output/expand_params/ \
    --embedding_name pos_embedding \
    --embedding_new_size 512

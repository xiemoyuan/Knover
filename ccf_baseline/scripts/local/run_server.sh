#!/bin/bash
set -eux

export CUDA_VISIBLE_DEVICES=0

python -u \
    ./run_server.py \
    --bot_name Shift_UniLM \
    --model UnifiedTransformer \
    --vocab_path ./package/dialog_en/vocab.txt \
    --do_lower_case false \
    --spm_model_file ./package/dialog_en/spm.model \
    --init_pretraining_params ./24L/Shift_UniLM \
    --do_generation true \
    --num_samples 20 \
    --config_path ./package/dialog_en/24L.json

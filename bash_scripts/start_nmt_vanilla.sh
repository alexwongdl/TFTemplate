#!/bin/bash

python MainEntry.py \
    --task=train_nmt \
    --is_training=True \
    --max_iter=200000 \
    --batch_size=16 \
    --learning_rate=0.01 \
    --decay_step=10000 \
    --decay_rate=0.9 \
    --input_dir=/home/recsys/hzwangjian1/learntf/TFTemplate/data/translate/subfiles \
    --model_dir=/home/recsys/hzwangjian1/learntf/nmt_vanilla_model \
    --save_model_freq=10000 \
    --summary_dir=/home/recsys/hzwangjian1/learntf/nmt_vanilla_summary \
    --summary_freq=100 \
    --print_info_freq=100 \
    --valid_freq=10000 \
    --init_scale=0.04 \
    --max_grad_norm=10 \
    --keep_prob=0.35 \
    --vocab_dim=200


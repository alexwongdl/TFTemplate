#!/bin/bash

python MainEntryPtb.py \
    --task=train_ptb \
    --is_training=True \
    --max_iter=200000 \
    --batch_size=20 \
    --learning_rate=1.0 \
    --decay_step=10000 \
    --decay_rate=0.9 \
    --input_dir=/home/recsys/hzwangjian1/learntf/TFTemplate/data/ptb \
    --save_model_dir=/home/recsys/hzwangjian1/learntf/ptb_model \
    --save_model_freq=10000 \
    --summary_dir=/home/recsys/hzwangjian1/learntf/ptb_summary \
    --summary_freq=100 \
    --print_info_freq=100 \
    --valid_freq=10000 \
    --init_scale=0.04 \
    --max_grad_norm=10 \
    --num_steps=35 \
    --keep_prob=0.35 \
    --vocab_size=10000 \
    --vocab_dim=200

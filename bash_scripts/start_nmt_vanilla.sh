#!/bin/bash

python MainEntryNMT_vanilla.py \
    --task=train_nmt \
    --is_training=True \
    --max_iter=200000 \
    --max_max_epoch=10 \
    --batch_size=16 \
    --learning_rate=1 \
    --decay_step=10000 \
    --decay_rate=0.99 \
    --input_dir=/data/hzwangjian1/tensorflow/nmt/subfiles_format \
    --save_model_dir=/home/recsys/hzwangjian1/learntf/nmt_vanilla_model \
    --save_model_freq=1000 \
    --summary_dir=/home/recsys/hzwangjian1/learntf/nmt_vanilla_summary \
    --summary_freq=100 \
    --print_info_freq=100 \
    --valid_freq=1000 \
    --init_scale=0.04 \
    --max_grad_norm=5 \
    --keep_prob=0.8 \
    --vocab_size=40000 \
    --vocab_dim=200 \
    --dict_one_path=/home/recsys/hzwangjian1/learntf/TFTemplate/data/translate/fr_dict \
    --dict_two_path=/home/recsys/hzwangjian1/learntf/TFTemplate/data/translate/en_dict


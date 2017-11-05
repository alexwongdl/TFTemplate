#!/bin/bash


python MainEntryMnist.py \
    --task=train_mnist \
    --is_training=True \
    --max_iter=200000 \
    --batch_size=16 \
    --learning_rate=0.01 \
    --decay_step=10000 \
    --decay_rate=0.9 \
    --input_dir=/home/recsys/hzwangjian1/learntf/MNIST \
    --save_model_dir=/home/recsys/hzwangjian1/learntf/mnist_model \
    --save_model_freq=10000 \
    --summary_dir=/home/recsys/hzwangjian1/learntf/mnist_summary



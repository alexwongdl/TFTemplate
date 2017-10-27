#!/bin/bash

python MainEntry.py \
    --task=train_mnist \
    --is_training=True \
    --max_iter=200000 \
    --batch_size=16 \
    --learning_rate=0.01 \
    --decay_step=10000 \
    --decay_rate=0.9 \
    --input_dir=../data/MNIST \
    --model_dir=../mnist_model \
    --summary_dir=../mnist_summary \

#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

dataset=organsmnist

accelerate launch train_jempp.py \
    --model yopo \
    --lr 0.1 \
    --optimizer sgd \
    --norm batch \
    --n_epochs 150 \
    --decay_epochs 50 100 125 \
    --px 1.0 \
    --pyx 1.0 \
    --l2 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 250 \
    --labels_per_class 35 \
    --dataset $dataset \
    --experiment_type equal_jempp_sgd \
    --enable_tracking \
    --multi_gpu \
    --calibrated \
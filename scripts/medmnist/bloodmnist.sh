#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

dataset=bloodmnist

accelerate launch train_jempp.py \
    --model yopo \
    --lr 0.1 \
    --optimizer sgd \
    --norm batch \
    --n_epochs 150 \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 250 \
    --labels_per_class 50 \
    --dataset $dataset \
    --experiment_type equal_jempp_sgd \
    --enable_tracking \
    --multi_gpu \
    --calibrated \
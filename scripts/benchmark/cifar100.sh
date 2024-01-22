#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

accelerate launch --main_process_port 29502 train_jempp.py \
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
    --query_size 2500 \
    --dataset cifar100 \
    --experiment_type active_calibrated \
    --calibrated \
    --enable_tracking \

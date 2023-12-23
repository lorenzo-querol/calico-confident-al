#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

accelerate launch train_jempp.py \
    --model yopo \
    --lr 0.1 \
    --optimizer sgd \
    --norm batch \
    --n_epochs 100 \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 2500 \
    --dataset mnist \
    --experiment_type active \
    --enable_tracking \
    --calibrated \
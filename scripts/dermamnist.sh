#!/bin/bash

export CUDA_VISIBLE_DEVICES="2,3,4,5"

accelerate launch train_jempp.py \
    --model yopo \
    --lr 0.0001 \
    --optimizer adam \
    --norm none \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 250 \
    --dataset dermamnist \
    --experiment_type baseline \
    # --calibrated \
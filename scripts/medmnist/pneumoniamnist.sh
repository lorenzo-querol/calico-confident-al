#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,3,4"

dataset=pneumoniamnist

accelerate launch train_jempp.py \
    --model yopo \
    --lr 0.1 \
    --optimizer sgd \
    --norm batch \
    --n_epochs 1 \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 250 \
    --labels_per_class 100 \
    --dataset $dataset \
    --experiment_type equal_jempp_sgd \
    --calibrated \
    # --enable_tracking \
#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"

# Only need to change the experiment_name and dataset

for seed_num in {1..5}
do
python test_jempp.py \
    --model yopo \
    --lr 0.0001 \
    --optimizer adam \
    --norm none \
    --n_epochs 150 \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 250 \
    --dataset pneumoniamnist \
    --experiment_type baseline \
    --experiment_name "pneumoniamnist_epoch_150_adam" \
    --seed $seed_num \
    # --calibrated \
    # --enable_tracking
done
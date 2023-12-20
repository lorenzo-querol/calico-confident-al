#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"

for seed_num in {1..5}
do
python test_jempp.py \
    --model yopo \
    --lr 0.0001 \
    --optimizer adam \
    --norm none \
    --n_epochs 50 \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --query_size 1000 \
    --dataset dermamnist \
    --experiment_type active \
    --experiment_name "dermamnist_epoch_50" \
    --calibrated \
    --seed $seed_num
    # --enable_tracking
done
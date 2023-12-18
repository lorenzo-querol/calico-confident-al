#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

for i in {5..5}
do
accelerate launch test_jempp.py \
    --model yopo \
    --lr 0.0001 \
    --optimizer adam \
    --norm none \
    --decay_epochs 50 100 125 \
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 20 \
    --in_steps 5 \
    --query_size 500 \
    --dataset bloodmnist \
    --experiment_type baseline \
    --experiment_name "2023-12-17_00-53-57_bloodmnist" \
    --seed $i \
    --enable_tracking \
    # --calibrated \
done
#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

for seed_num in {5..5}
do
accelerate launch test_jempp.py \
    --query_size 500 \
    --dataset bloodmnist \
    --experiment_name "2023-12-17_00-53-57_bloodmnist" \
    --seed $seed_num \
    --enable_tracking
done
#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

for d in organsmnist
do
    for exp_type in equal_jempp_sgd
    do
        for i in {0..4}
        do 
            python test_jempp.py \
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
                --labels_per_class 35 \
                --dataset $d \
                --experiment_name $d \
                --experiment_type $exp_type \
                --seed $i
        done
    done
done
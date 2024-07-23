#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

for d in organcmnist bloodmnist
do
    for exp_type in active_softmax
    do
        for i in {0..0}
        do 
            python test_jempp.py \
                --model yopo \
                --lr 0.1 \
                --optimizer sgd \
                --norm batch \
                --n_epochs 150 \
                --decay_epochs 50 100 125 \
                --px 1.0 \
                --pyx 1.0 \
                --l2 0.0 \
                --n_steps 10 \
                --in_steps 5 \
                --query_size 250 \
                --dataset $d \
                --experiment_name $d \
                --experiment_type $exp_type \
                --seed $i
        done
    done
done
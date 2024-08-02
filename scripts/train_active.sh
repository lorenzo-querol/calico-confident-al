#!/bin/bash

DATASET=$1
export CUDA_VISIBLE_DEVICES="1"

python train_jempp.py \
    --model yopo \
    --lr 0.1 --optimizer sgd --norm batch --decay_epochs 50 100 125 \
    --n_epochs 150 \
    --px 0.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --dataset $DATASET --query_size 250 \
    --exp_name active --enable_tracking
    
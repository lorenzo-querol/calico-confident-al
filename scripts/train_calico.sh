#!/bin/bash

DATASET=$1
export CUDA_VISIBLE_DEVICES=$2

python train_active.py \
    --query_size 250 \
    --lr 0.1 --decay_epochs 50 100 125 --optim sgd --norm batch \
    --n_steps 10 --in_steps 5 \
    --n_epochs 150 --batch_size 64 \
    --px 1.0 --pyx 1.0 --l2 0.0 \
    --dataset $DATASET --exp_name 'calico'
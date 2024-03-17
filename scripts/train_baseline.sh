#!/bin/bash

DATASET=$1
EXP_NAME=$2
export CUDA_VISIBLE_DEVICES=$3

# Don't forget to change the query size for benchmark or medmnist
# For benchmark, use 40000
# For medmnist, use 4000

python train_jempp.py \
    --model yopo \
    --lr 0.1 --decay_epochs 60 120 180 --optim sgd --norm batch \
    --n_epochs 200 --batch_size 128 \
    --px 0.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --query_size 40000 --sample_method random \
    --dataset $DATASET --exp_name $4 \
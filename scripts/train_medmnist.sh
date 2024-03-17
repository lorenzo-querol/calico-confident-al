#!/bin/bash

DATASET=$1
PX=$2
EXP_NAME=$3
export CUDA_VISIBLE_DEVICES=$4

python train_jempp.py \
    --model yopo \
    --lr 0.01 --decay_epochs 25 --optim sgd --norm batch --warmup_iters -1  \
    --n_epochs 50 --batch_size 128 \
    --px $PX --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --query_size 250 --sample_method random \
    --dataset $DATASET --exp_name $4 \
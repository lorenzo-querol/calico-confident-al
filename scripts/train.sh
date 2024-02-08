#!/bin/bash

dataset=$1
px=$2
gpu=$3

export CUDA_VISIBLE_DEVICES=$gpu

python train_jempp.py \
    --n_epochs 200 --decay_epochs 60 120 180 \
    --n_steps 10 \
    --px $px --rho 0.2 \
    --lr 0.01 --optim sgd --norm batch \
    --dataset $dataset \
    --exp_name $dataset \
    --sam \
    # --enable_log
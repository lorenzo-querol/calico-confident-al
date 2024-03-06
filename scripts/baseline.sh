#!/bin/bash

dataset=$1
gpu=$2

export CUDA_VISIBLE_DEVICES=$gpu

python train_jempp.py \
    --n_epochs 200 --decay_epochs 60 120 180 \
    --px 0.0 --n_steps 5 \
    --lr 0.1 --optim sgd --norm batch \
    --dataset $dataset --exp_name $dataset
    
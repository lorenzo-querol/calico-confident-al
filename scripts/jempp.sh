#!/bin/bash

dataset=$1
gpu=$2

export CUDA_VISIBLE_DEVICES=$gpu

python train_jempp.py \
    --n_epochs 200 --decay_epochs 60 120 180 \
    --n_steps 10 \
    --batch_size 128 \
    --lr 0.1 --optim sgd --norm batch  \
    --dataset $dataset --exp_name $dataset \
    # --sam --rho 0.2 \
    # --test --ckpt_path checkpoints/bloodmnist/train/v_1 \

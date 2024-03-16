#!/bin/bash

DATASET=$1
PX=$2
export CUDA_VISIBLE_DEVICES="0"

python train_jempp.py \
    --model yopo \
    --lr 0.01 --optim sgd --norm batch --warmup_iters -1 \
    --n_epochs 50 --batch_size 128 \
    --decay_epochs 25 100 125 \
    --px $PX --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --query_size 250 --sample_method random \
    --labels_per_class -1 \
    --dataset $DATASET --sigma 0.03 \
    --test --ckpt_dir logs/bloodmnist/v_0/checkpoints
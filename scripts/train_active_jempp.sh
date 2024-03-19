#!/bin/bash

DATASET=$1
export CUDA_VISIBLE_DEVICES=$2

# Don't forget to change the query size and lr for Benchmark or MedMNIST.
# Benchmark (CIFAR10, etc.): query_size 2500
# MedMNIST: query_size 250 

python train_jempp.py \
    --query_size 250 --lr 0.001 \
    --model yopo --norm batch \
    --decay_epochs 25 --optim sgd --warmup_iters -1 \
    --n_epochs 50 --batch_size 128 \
    --px 1.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --sample_method random \
    --dataset $DATASET --exp_name 'active-jempp-v2'
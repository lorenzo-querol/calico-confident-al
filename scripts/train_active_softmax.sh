#!/bin/bash

DATASET=$1
export CUDA_VISIBLE_DEVICES=$2

# Don't forget to change the query size and lr for Benchmark or MedMNIST.
# Benchmark (CIFAR10, etc.): query_size 2500
# MedMNIST: query_size 250 

python train_jempp.py \
    --query_size 250 --lr 0.1 \
    --model yopo --norm batch \
    --decay_epochs 50 100 125 --optim sgd \
    --n_epochs 150 --batch_size 64 \
    --px 0.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --sample_method random \
    --dataset $DATASET --exp_name 'active-softmax'
#!/bin/bash

# Define the datasets here. Easier to run these since it's less computationally expensive than JEM++
DATASETS=('bloodmnist' 'dermamnist' 'organsmnist' 'organcmnist')
export CUDA_VISIBLE_DEVICES=$1

# Don't forget to change the query size and lr for Benchmark or MedMNIST.
# Benchmark (CIFAR10, etc.): query_size 40000, lr 0.1
# MedMNIST: query_size 4000, lr 0.01

for DATASET in "${DATASETS[@]}"
do
    python train_jempp.py \
        --query_size 4000 --lr 0.1 \
        --model yopo --norm batch \
        --decay_epochs 60 120 160 --optim sgd \
        --n_epochs 200 --batch_size 128 \
        --px 0.0 --pyx 1.0 --l2 0.0 \
        --n_steps 10 --in_steps 5 \
        --sample_method random \
        --dataset $DATASET --exp_name baseline-softmax
done
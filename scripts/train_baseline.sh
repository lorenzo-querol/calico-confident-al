#!/bin/bash

# Define the datasets here. Easier to run these since it's less computationally expensive than JEM++
DATASETS=('bloodmnist' 'organcmnist' 'organsmnist' 'pneumoniamnist' 'dermamnist')
export CUDA_VISIBLE_DEVICES=$1

for DATASET in "${DATASETS[@]}"
do
    python train_jempp.py \
        --query_size 250 --lr 0.1 \
        --model yopo --norm batch \
        --decay_epochs 50 100 125 --optim sgd \
        --n_epochs 150 --batch_size 64 \
        --px 1.0 --pyx 1.0 --l2 0.0 \
        --n_steps 10 --in_steps 5 \
        --sample_method random \
        --dataset $DATASET --exp_name baseline-softmax
done
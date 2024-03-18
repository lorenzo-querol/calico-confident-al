#!/bin/bash

# Define the datasets
DATASETS=("bloodmnist" "dermamnist" "organcmnist" "organsmnist" "pneumoniamnist")

EXP_NAME="baseline-softmax"
export CUDA_VISIBLE_DEVICES=$1

# Loop through the datasets
for DATASET in "${DATASETS[@]}"
do
    # Don't forget to change the query size for benchmark or medmnist
    # For benchmark, use 40000
    # For medmnist, use 4000

    python train_jempp.py \
        --model yopo \
        --lr 0.01 --decay_epochs 60 120 160 --optim sgd --norm batch \
        --n_epochs 200 --batch_size 128 \
        --px 0.0 --pyx 1.0 --l2 0.0 \
        --n_steps 10 --in_steps 5 \
        --query_size 4000 --sample_method random \
        --dataset $DATASET --exp_name ${EXP_NAME}
done
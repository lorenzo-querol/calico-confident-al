#!/bin/bash

DATASET=$1
LABELS_PER_CLASS=$2
export CUDA_VISIBLE_DEVICES=$3

# NOTE The labels_per_class for each dataset are as follows:
# bloodmnist: 4000, labels_per_class 50
# organcmnist: 3850, labels_per_class 35
# organsmnist: 3850, labels_per_class 35
# pneumoniamnist: 2000, labels_per_class 100

python train_jempp.py \
    --query_size 250 --lr 0.001 \
    --model yopo --norm batch \
    --decay_epochs 25 --optim sgd --warmup_iters -1 \
    --n_epochs 50 --batch_size 128 \
    --px 1.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --sample_method equal --labels_per_class ${LABELS_PER_CLASS} \
    --dataset $DATASET --exp_name 'equal-jempp-v2'
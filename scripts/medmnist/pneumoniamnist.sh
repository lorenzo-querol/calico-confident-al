#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

dataset=pneumoniamnist

python train_jempp.py \
    --model yopo \
    --lr 0.1 --optim sgd --norm batch \
    --n_epochs 150  \
    --decay_epochs 50 100 125 \
    --px 1.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --query_size 250 --sample_method random \
    --labels_per_class -1 \
    --dataset $dataset --sigma 0.0 \
    --experiment_type random-independent_softmax_sgd \
    # --enable_tracking \

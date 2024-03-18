#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python train_jempp.py \
    --query_size -1 --lr 0.1 \
    --model yopo \
    --decay_epochs 25 --optim sgd --norm batch --warmup_iters -1  \
    --n_epochs 50 --batch_size 128 \
    --px 1.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --sample_method random \
    --test

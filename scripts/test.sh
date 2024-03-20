#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python train_jempp.py \
    --query_size -1 --lr 0.1 \
    --model yopo --norm batch \
    --decay_epochs 50 100 125 --optim sgd \
    --n_epochs 150 --batch_size 64 \
    --px 1.0 --pyx 1.0 --l2 0.0 \
    --n_steps 10 --in_steps 5 \
    --sample_method random \
    --test --log_dir runs --test_dir runs_results

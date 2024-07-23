#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

QUERY_SIZE=13940
DATASET=organsmnist
N_EPOCHS=150
OPTIMIZER=sgd
EXPERIMENT_NAME=${DATASET}_epoch_${N_EPOCHS}_${OPTIMIZER}

python class_dist.py \
    --model yopo \
    --lr 0.1 \
    --optimizer $OPTIMIZER \
    --norm batch \
    --n_epochs $N_EPOCHS \
    --decay_epochs 50 100 125 \
    --px 1.0 \
    --pyx 1.0 \
    --l2 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --labels_per_class 0 \
    --dataset $DATASET \
    --query_size $QUERY_SIZE \
    --experiment_name active_calibrated \
    --experiment_type active_calibrated \
    --seed 1
#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

QUERY_SIZE=250
DATASET=pneumoniamnist
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
    --p_x_weight 1.0 \
    --p_y_x_weight 1.0 \
    --l2_weight 0.0 \
    --n_steps 10 \
    --in_steps 5 \
    --dataset $DATASET \
    --experiment_name active_calibrated \
    --experiment_type active_calibrated \
    --seed 1
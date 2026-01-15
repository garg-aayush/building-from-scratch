#!/bin/bash

# BATCH_PER_EI=512
# NUM_ROLLOUTS=4
# WANDB_RUN_NAME="run_D${BATCH_PER_EI}_G5_R${NUM_ROLLOUTS}"

NUM_EPOCHS=2
for NUM_ROLLOUTS in 4; do
    for BATCH_PER_EI in 512 1024; do
        WANDB_RUN_NAME="run_Ep${NUM_EPOCHS}_D${BATCH_PER_EI}_G5_R${NUM_ROLLOUTS}"
        uv run train.py --batch_per_ei $BATCH_PER_EI --num_rollouts $NUM_ROLLOUTS --wandb_run_name $WANDB_RUN_NAME --num_epochs $NUM_EPOCHS
    done
done
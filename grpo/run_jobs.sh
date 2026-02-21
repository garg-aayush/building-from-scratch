#!/bin/zsh


# # -------------------------------------------------------------#
# # LR sweep
# # -------------------------------------------------------------#
# CONFIGS=(
#     # "configs/lr_sweep/lr_1e-6.yaml"
#     # "configs/lr_sweep/lr_3e-6.yaml"
#     # "configs/lr_sweep/lr_1e-5.yaml"
#     # "configs/lr_sweep/lr_1.5e-5.yaml"
#     # "configs/lr_sweep/lr_3e-5.yaml"
#     # "configs/lr_sweep/lr_1e-4.yaml"
# )

# OUTPUT_DIR="/results/lr_sweep"

# -------------------------------------------------------------#
# Baselines
# -------------------------------------------------------------#
CONFIGS=(
    # "configs/baselines/no_baseline.yaml"
    # "configs/baselines/reinforce_baseline.yaml"
    "configs/baselines/no_baseline_lr1.5e-5.yaml"
)
OUTPUT_DIR="/results/baselines"

for config in "${CONFIGS[@]}"; do
    echo "Running $config"
    # cmd="modal run --detach train_on_modal.py \
    #     --config $config \
    #     --output-dir /results/lr_sweep/$(basename $config) \
    #     --spawn"
    # echo $cmd, remove .yaml from the config name
    dir_name=$(basename $config .yaml)
    # echo $dir_name
    modal run --detach train_on_modal.py --config $config --output-dir $OUTPUT_DIR/$dir_name --spawn
done
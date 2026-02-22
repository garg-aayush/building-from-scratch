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

# # -------------------------------------------------------------#
# # Baselines
# # -------------------------------------------------------------#
# CONFIGS=(
#     # "configs/baselines/no_baseline.yaml"
#     # "configs/baselines/reinforce_baseline.yaml"
#     "configs/baselines/no_baseline_lr1.5e-5.yaml"
# )
# OUTPUT_DIR="/results/baselines"

# # -------------------------------------------------------------#
# # Length Normalization
# # -------------------------------------------------------------#
# CONFIGS=(
#     "configs/length_normalization/len_norm_mean.yaml"
#     "configs/length_normalization/len_norm_constant.yaml"
#     "configs/length_normalization/len_norm_microbatch.yaml"
# )
# OUTPUT_DIR="/results/length_normalization"

# # -------------------------------------------------------------#
# # Std Dev Normalization
# # -------------------------------------------------------------#
# CONFIGS=(
#     "configs/std_dev/std_dev.yaml"
# )
# OUTPUT_DIR="/results/std_dev"

# # -------------------------------------------------------------#
# # Off-Policy Sweep
# # -------------------------------------------------------------#
# CONFIGS=(
#     "configs/off_policy_sweep/e1_tb256_ga64.yaml"
#     "configs/off_policy_sweep/e1_tb128_ga32.yaml"
#     "configs/off_policy_sweep/e2_tb256_ga64.yaml"
#     "configs/off_policy_sweep/e2_tb128_ga32.yaml"
#     "configs/off_policy_sweep/e4_tb256_ga64.yaml"
#     "configs/off_policy_sweep/e4_tb128_ga32.yaml"
#     "configs/off_policy_sweep/e4_tb64_ga16.yaml"
# )
# OUTPUT_DIR="/results/off_policy_sweep"


# -------------------------------------------------------------#
# Off-Policy Sweep
# -------------------------------------------------------------#
CONFIGS=(
    "configs/off_policy_sweep/full_e1_tb256_ga64.yaml"
    "configs/off_policy_sweep/full_e1_tb128_ga32.yaml"
    "configs/off_policy_sweep/full_e2_tb256_ga64.yaml"
)
OUTPUT_DIR="/results/off_policy_sweep"


for config in "${CONFIGS[@]}"; do
    echo "Running $config"
    dir_name=$(basename $config .yaml)
    modal run --detach train_on_modal.py --config $config --output-dir $OUTPUT_DIR/$dir_name --spawn
done
# Supervised Fine-Tuning Track

This directory contains my Supervised Fine-Tuning (SFT) implementation, training experiments,and relevant evaluation scripts. I’m loosely following the SFT portion of Stanford's [CS336](https://stanford-cs336.github.io/spring2025/) course's [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment) (both the main and supplementary parts). Once the core SFT and evaluation code are ready, I plan to explore additional experiments and potential improvements.

## To Do
- [x] Setup vLLM for offline batched inference
    - Fairly simple to install and setup at least with CUDA 12.8
    - Refer to [Notes.md](Notes.md) for more details on the installation and usage
- [x] Create the training and validation datasets
    - [x] Create the train and validation splits using the `hiyouga/math12k` dataset
    - [x] Batch infer the train data for reasoning traces using the `deepseek-v3p1-terminus`
    - [x] Build the final SFT dataset
    > Refer to [Notes.md](Notes.md) for more details on the dataset creation pipeline
- [x] Run baseline evaluation
    - [x] Run `Qwen/Qwen2.5-Math-1.5B` on the validation set (val.jsonl) to calculate the baseline accuracy
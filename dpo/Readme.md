# Direct Preference Optimization (DPO) from scratch

This folder contains **from-scratch implementation** of Direct Preference Optimization (DPO), loosely following Stanford's [CS336 Supplementary Assignment 5](https://github.com/heng380/cs336_assignment-5/blob/main/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf).

## Notes
- Create training dataset:
    - As per the assignment, the training data comes from Anthropic's [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset.
    - Basically, we filter the data to only keep the single-turn conversations and create a dataset of around ~51K examples.
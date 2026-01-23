# Expert Iteration from Scratch

This folder contains my from-scratch implementation of Expert Iteration for math reasoning, following Stanford's [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment). Expert Iteration, unlike SFT, does not require expensive human-annotated reasoning traces. Instead, the model generates candidate solutions, filters for correct ones and trains on its own successful attempts.

> Best: **47.1% validation accuracy** (up from 2.9% baseline) with D=1024, R=4, 2 epochs config

![Expert Iteration Best Results](plots/ei_multiepoch_plot.png)

> Note: I have also written a detailed blogpost about the experiments and analysis. You can find it here: [Expert Iteration for Math Reasoning](https://huggingface.co/blog/garg-aayush/expert-iteration-math-reasoning).


## Table of Contents
- [What is Expert Iteration](#what-is-expert-iteration)
- [Experiments](#experiments)
  - [Model, Data and Checkpoints](#model-data-and-checkpoints)
  - [Finding the Right Learning Rate](#finding-the-right-learning-rate)
  - [Single vs Multiple Reasoning Traces](#single-vs-multiple-reasoning-traces)
  - [Exploring D and R Grid](#exploring-d-and-r-grid)
  - [Multi-Epoch Training](#multi-epoch-training)
- [Comparison to SFT](#comparison-to-sft)
- [Folder Structure](#folder-structure)


## What is Expert Iteration

Expert Iteration was first introduced by [Anthony et. al. (2017)](https://arxiv.org/abs/1705.08439) for game-playing AI. The [Self-taught Reasoner (STaR)](https://arxiv.org/abs/2203.14465) applies this idea to LLM reasoning: at each iteration, prompt the model to generate rationales, filter to keep only correct answers, finetune on the filtered set and repeat.

I see Expert Iteration as **SFT on self-generated, filtered data, repeated iteratively**:

![Expert Iteration Diagram](plots/expert-iteration-diagram.png)

## Experiments

### Model, Data and Checkpoints
- **Model**: [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) (base, not instruction-tuned)
- **Training and val data**: [sft-cs336-assign5-datasets](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/tree/main/sft-reason)

    | File | Description |
    |------|-------------|
    | [train.jsonl](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/blob/main/sft-reason/sft_gpt-oss-120b_filtered.jsonl) | ~3.5K training problems with ground-truth answers |
    | [val.jsonl](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/blob/main/sft-reason/val.jsonl) | 5K validation problems from CS336 Assignment 5 |

- **Finetuned Checkpoints**: [cs336_exp-iter_exps](https://huggingface.co/garg-aayush/cs336_exp-iter_exps/)

- **Training Logs**: [wandb expert-iter](https://wandb.ai/garg-aayush/expert-iter)

### Key Hyperparameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `batch_per_ei` | D | number of questions sampled per iteration |
| `num_rollouts` | R | number of outputs generated per question |
| `num_ei` | G | number of expert iteration steps (fixed at 5) |

Each iteration samples D questions, generates R candidate solutions per question (D × R total rollouts), filters for correct answers and finetunes on the filtered set.


### Finding the Right Learning Rate

The number of filtered examples varies significantly across iterations in exp-iter with early iterations filtering fewer correct solutions. This makes learning rate selection important.

![Learning Rate Sweep](plots/lr_sweep_ei_acc.png)

The best way to handle this is an **adaptive scheme** that scales both learning rate and batch size:

| Filtered Examples | Batch Size | Learning Rate |
|-------------------|------------|---------------|
| < 24              | 8          | 3.5e-5        |
| 24-128            | 32         | 5e-5          |
| > 128             | 64         | 7e-5          |


### Single vs Multiple Reasoning Traces

This experiment compares the effect of single vs multiple rollouts on validation accuracy.

![Sampling Strategy Comparison](plots/sampling_strategy_ei_acc.png)

Here, single-trace refers to keeping only one trace per question, while multi-trace refers to keeping all correct traces per question in sampling. Multi-trace sampling reaches higher accuracy than single-trace as diverse reasoning paths provide a richer training signal. 

### Exploring D and R Grid

Ran a grid of experiments varying D (batch_per_ei) and R (num_rollouts):

![D and R Grid Results](plots/ei_grid_plot.png)

- Best configuration is D=1024, R=4 achieving **41.7%** accuracy
- Increasing D does not always guarantee better performance; D=2048 underperforms D=1024
- Increasing R from 2 to 4 improves accuracy across all D values. More rollouts means higher solve probability and more diverse reasoning traces

### Multi-Epoch Training

Also, tried to improve the accuracy by training for 2 epochs per iteration with the best configurations:

![Multi-Epoch Results](plots/ei_multiepoch_plot.png)

With large $R$, the filtered dataset is diverse enough that training for multiple epochs extracts more signal without memorizing. The best accuracy is achieved with D=1024, R=4, 2 epochs with **47.1%** accuracy in comparison to **41.7%** with 1 epoch.


## Comparison to SFT

Comparing to my [previous SFT experiments](https://huggingface.co/blog/garg-aayush/building-sft-from-ground-up) using the same model and data:

| Method | Configuration | Validation Accuracy |
|--------|---------------|---------------------|
| Baseline | Untrained Qwen2.5-Math-1.5B | 2.9% |
| **Expert Iteration (best)** | D=1024, R=4, 2 epochs | **47.1%** |
| **SFT (best)** | Filtered GPT traces, 2 epochs | **53.4%** |


This gap is due to:
1. **Data quality**: SFT trains on GPT-OSS-120B traces with sophisticated problem-solving strategies. Expert Iteration trains on the model's own (less refined) outputs
2. **Rationalization**: The STaR paper shows that providing answer hints for failed problems accelerates learning. This implementation doesn't use rationalization

## Folder Structure
```
expert-iteration/
├── train_exp_iter.py       # Main Expert Iteration training script
├── evaluate.py             # Evaluation script for checkpoints
├── run_exp_iter.sh         # Shell script for running experiments
├── utils/
│   ├── dataloader.py       # Data loading functions
│   ├── drgrpo_grader.py    # Answer grading logic
│   ├── helper_fns.py       # Helper functions
│   ├── post_train.py       # Post-training utilities
│   └── upload_to_hf.py     # HuggingFace upload utility
├── data/                   # Training/validation data
├── play-scripts/           # Plotting and analysis scripts
├── plots/                  # Plots and analysis results
└── Readme.md               # This file
```

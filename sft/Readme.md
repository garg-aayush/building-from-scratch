# Supervised finetuning

This directory contains my Supervised finetuning (SFT) implementation, training experiments,and relevant evaluation scripts. I’m loosely following the SFT portion of Stanford's [CS336](https://stanford-cs336.github.io/spring2025/) course's [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment) (both the main and supplementary parts). Once the core SFT and evaluation code are ready, I plan to explore additional experiments and potential improvements.

## Datasets and Checkpoints
For now I have uploaded all the training and evaluation datasets to the Hugging Face Hub. 

### SFT Instruction finetuning

**Datasets:**
The instruction-tuning dataset is available in the [sft-instruct](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/tree/main/sft-instruct) folder of HF's repo `sft-cs336-assign5-datasets`.
| File | Description |
|------|-------------|
| `train.jsonl` | Training dataset (~200k examples) |
| `test.jsonl` | Validation dataset for intermediate evaluation |
| `sample_train.jsonl` | 1k-example subsample for debugging |

> **Note:** If any eval data files are missing locally, download them from the [eval](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/tree/main/sft-instruct/eval) subfolder (contains GSM8K, MMLU, Simple Safety Tests, Alpaca Eval).

**Training checkpoints:**
| Run | Checkpoint |
|-----|------------|
| With prompt masking (`run_mask`) | [garg-aayush/llama31-8b-sft-mask](https://huggingface.co/garg-aayush/llama31-8b-sft-mask) |
| Without prompt masking (`run_nomask`) | [garg-aayush/llama31-8b-sft-nomask](https://huggingface.co/garg-aayush/llama31-8b-sft-nomask) |

**Training logs** can be found in the wandb's [sft_instruct](https://wandb.ai/garg-aayush/sft_instruct) project.

### SFT Reasoning finetuning
The reasoning SFT datasets (for `Qwen/Qwen2.5-Math-1.5B`) are available in the [sft-reason](https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets/tree/main/sft-reason) folder of HF's repo `sft-cs336-assign5-datasets`.

**Datasets:**
| File | Examples | Description |
|------|----------|-------------|
| `sft_gpt-oss-120b_filtered.jsonl` | 3,496 | Recommended - Contains only correct reasoning traces |
| `sft_gpt-oss-120b.jsonl` | 4,836 | Full dataset with both correct and incorrect answers |
| `val.jsonl` | ~5K | Validation set from CS336 Assignment 5 for evaluation |

> For details on how these datasets were created (batch inference pipeline, filtering, etc.), see [Notes.md](Notes.md#dataset-creation-pipeline).

**Training checkpoints:**
| Run | Checkpoint |
|-----|------------|
| Trained on full training data (`run_all`) | [garg-aayush/qwen-2.5-math-sft-all](https://huggingface.co/garg-aayush/qwen-2.5-math-sft-all) |
| Trained on filtered training data (`run_filtered`) | [garg-aayush/qwen-2.5-math-sft-filtered](https://huggingface.co/garg-aayush/qwen-2.5-math-sft-filtered) |
| Trained on filtered training data with no per-token loss normalization (`run_filtered-res-len`) | [garg-aayush/qwen-2.5-math-sft-filtered-res-len](https://huggingface.co/garg-aayush/qwen-2.5-math-sft-filtered-res-len) |
| Trained on filtered training data for 2 epochs (`run_filtered-2epoch`) | [garg-aayush/qwen-2.5-math-sft-filtered-2epoch](https://huggingface.co/garg-aayush/qwen-2.5-math-sft-filtered-2epoch) |

**Training logs** can be found in the wandb's [sft](https://wandb.ai/garg-aayush/sft) project.

## Folder Structure
```
sft/
├── train_sft_instruct.py       # Instruction-tuning training
├── train_sft_reason.py         # Reasoning SFT training
├── evaluate_instruct.py        # Instruction-tuning evaluation
├── evaluate_reason.py          # Reasoning evaluation
├── results/                    # SFT experiments results
├── utils/
│   ├── dataloader.py           # SFT-Reasoning data loading fns
│   ├── drgrpo_grader.py        # SFT-Reasoning answer-grading logic
│   ├── helper_fns.py           # helper fns
│   ├── instruct_dataset.py     # SFT-Instruction finetuning dataset/dataloader class & fns
│   ├── instruct_measures.py    # SFT-Instruction finetuning eval metrics
│   └── post_train.py           # SFT helper functions
├── data/
│   ├── eval_configs/           # YAML configs for SFT-Instruction finetuning evaluation datasets
│   ├── eval_data/              # Evaluation dataset files
│   └── *.prompt                # Prompt templates for train/evals
├── play-scripts/
│   ├── eval-scripts/           # Individual SFT-Instruction finetuning evaluation scripts
│   └── plot-scripts/           # vibe-coded plotting scripts
├── Notes.md                    # Development notes
├── Readme.md                   # This file
└── requirements.txt            # Python dependencies
```

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
- [x] Write all sft helper functions as per [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_alignment.pdf)
    - [x] Write and test the following helper functions:
        - `tokenize_prompt_and_output`,`compute_entropy`, `get_response_log_probs`, `masked_normalize`, `sft_microbatch_train_step`
- [x] Regenerate the SFT training dataset using the `gpt-oss-120b` model 
- [x] Write the SFT training code
    - [x] Write the minimal SFT script (without evaluation and logging)
    > Note: I updated the sft_microbatch_train_step function to support per-token loss calculation. Basically now you can calculate the loss as per-token loss as well as the sum over the sequence dimension. You actually see stable training with per-token loss with acceptable loss and gradient norms. I will talk more about this when I am done with writing the code and is running the experiments.
    - [x] Add wandb logging
    - [x] Add vllm-based intermediate evaluation
        - Now, we can evaluate the model on the validation set using the vLLM model. Note, there are two had to make to vLLM model based evaluation to work. Please see [Notes.md](Notes.md) for more details.    
    - [x] log intermediate evaluation examples to jsonl files and to wandb
    - [x] evaluate the model on the val data and log eval metrics like loss, tok_entropy etc
- [x] Run the SFT training experiments for Qwen/Qwen2.5-Math-1.5B
- [x] Compare different runs accuracy on full validation data

## To Do (Supplementary Assignment 5)
- [-] Write evaluation scripts and evaluate the `Llama-3.1-8B` model (baseline)
    - [x] Write the evaluation script for the mmlu dataset
    - [x] Write the evaluation script for the gsm8k dataset
    - [x] Write the evaluation script for the alpaca_eval dataset
    - [x] Write the evaluation script for the simple_safety_tests dataset
    - [x] Create a unified evaluation script for all eval datasets
        - [x] Move the individual evaluation scripts to the `play-scripts/eval-scripts` directory
- [x] Write the sft training code for instruction-finetuning
    - [x] Add the dataset creating and loader script
    - [x] Add the instruction-finetuning sft training script
- [-] Run the SFT training experiments for `Llama-3.1-8B`
    - [x] With params mentioned in the assignment and no prompt masking
    - [x] with prompts masking
    - [ ] if time and costs permit, search for better params 
- [x] Compute the instruction-finetuning SFT evaluation metrics
    - [x] Compute the accuracy metrics for no prompt masking exp.
    - [x] Compute the accuracy metrics for prompt masking exp.
- [x] Plot and analyze the different sft training results
    - [x] Plot and analyze the reasoning SFT training results for the `Qwen/Qwen2.5-Math-1.5B` model
    - [x] Plot and analyze the instruction-finetuning SFT training results for the `Llama-3.1-8B` model 
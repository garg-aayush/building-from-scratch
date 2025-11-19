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
- [x] Write all sft helper functions as per [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_alignment.pdf)
    - [x] Write and test the following helper functions:
        - `tokenize_prompt_and_output`,`compute_entropy`, `get_response_log_probs`, `masked_normalize`, `sft_microbatch_train_step`
- [x] Regenerate the SFT training dataset using the `gpt-oss-120b` model 
- [-] Write the SFT training code
    - [x] Write the minimal SFT script (without evaluation and logging)
    > Note: I updated the sft_microbatch_train_step function to support per-token loss calculation. Basically now you can calculate the loss as per-token loss as well as the sum over the sequence dimension. You actually see stable training with per-token loss with acceptable loss and gradient norms. I will talk more about this when I am done with writing the code and is running the experiments.
    - [x] Add wandb logging
    - [x] Add vllm-based intermediate evaluation
        - Now, we can evaluate the model on the validation set using the vLLM model. Note, there are two had to make to vLLM model based evaluation to work. Please see [Notes.md](Notes.md) for more details.    
    - [x] log intermediate evaluation examples to jsonl files and to wandb
    - [x] evaluate the model on the val data and log eval metrics like loss, tok_entropy etc
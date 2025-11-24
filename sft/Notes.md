## vLLM Installation
- Follow the steps mentioned in [vLLM's GPU installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- After installation, you can run [vllm_ex_offline_batching.py](play-scripts/vllm_ex_offline_batching.py) script
- Additional notes:
    - vLLM by default downloads the model from the Hugging Face Hub
    - For clarity on parameters, refer to:
        - `SamplingParams` [documentation](https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams)
        - `LLM` [documentation](https://docs.vllm.ai/en/latest/api/vllm/#vllm.LLM)
    - In case OOMs error during model loading, 
       - set the `max_model_len` parameter to a smaller value say 2048/4096 (ensures KV cache is not too large).
       - set `max_num_seqs` parameter to a smaller value say 16/32/64 (ensures batch size is not too large).
       - set `gpu_memory_utilization` parameter to a smaller value say 0.2/0.3... (ensures the model fits in the GPU memory).

## Dataset Creation Pipeline

I created the three datasets can be used both SFT and GRPO training (in later experiments):

### Step 1: Creating the Train and Validation Splits
**Script**: [make_datasets.py](play-scripts/make_datasets.py)

- I started with `math_results.jsonl` that contains the validation data info used in assignment-5 of CS336 course and extracted the problems and expected answers to create my validation dataset: **val.jsonl**
- For the training data, I loaded the `hiyouga/math12k` dataset (combining both train and test splits), then filtered out any problems that appeared in the validation set to avoid data leakage. This gave me the training dataset: **train.jsonl**.

### Step 2: Batch Inference for Reasoning Traces
**Script**: [data4batch_infer.py](play-scripts/data4batch_infer.py)

- **train_data_4_batchinfer.jsonl**: I formatted each training problem with a prompt asking the model to provide reasoning and put the final answer in `\boxed{}` format. This created `train_data_4_batchinfer.jsonl`, which I then batch infered using the `deepseek-v3p1-terminus` model available via [Fireworks AI](https://fireworks.ai/).

### Step 3: Building the Final SFT Dataset
**Script**: [extract_answers_from_batch.py](play-scripts/extract_answers_from_batch.py)

Finally, I processed the batch inference outputs to create **sft.jsonl** using the following logic: 
- I filtered for only the successfully completed responses (where finish_reason was `stop`) and extracted the reasoning traces and answers from the model outputs using regex.
- Finally, after a bit of cleaning I created the final SFT dataset **sft.jsonl** with the following fields: `problem`, `reasoning_trace`, `extracted_answer`, and `expected_answer`.

### Update
- While writing the SFT training scripts, I realized the earlier `deepseek-v3p1-terminus` traces were inadequate in both format and answer quality. I regenerated them in the correct format and with correct answers using the `gpt-oss-120b` model. The updated datasets are:
  - `data/sft_gpt-oss-120b.jsonl`: 4,836 examples containing both correct and incorrect answers
  - `data/sft_gpt-oss-120b_filtered.jsonl`: 3,496 examples containing only correct answers


## Run baseline evaluation

I ran a baseline evaluation of the `Qwen/Qwen2.5-Math-1.5B` model using the `evaluate.py` script. It uses vLLM for running the model for evaluation and grading logic defined in `utils/drgrpo_grader.py`, following the evaluation criteria outlined in [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_alignment.pdf).

The baseline accuracy metrics are stored in `data/baseline_results.jsonl`:

| Metric           | Value    |
| :--------------- | :------- |
| Average Accuracy | 0.0288   |
| Average Format Accuracy | 0.1438   |

## Train the SFT model
### Write the helper functions first
As a first task, I wrote and tested the helper functions that we need for SFT training.
    - `tokenize_prompt_and_output`: Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for the prompt and padding tokens.
    - `compute_entropy`: Compute the entropy of the next token predictions.
    - `get_response_log_probs`: Get the response log-probs and the token entropy.
    - `masked_normalize`: Sum over a dimension and normalize by a constant, considering only the elements with mask value 1.
    - `sft_microbatch_train_step`: Train a single microbatch of data.
Basically, I followed the same approach and logic as in the [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_alignment.pdf) to write the helper functions.

### Add configurable loss normalization
I added configurable loss normalization in `sft_microbatch_train_step` via the `per_token_loss` flag to handle variable-length sequences. This is different from the original assignment where we only had the sum over the sequence dimension normalization. Without per-token normalization, longer sequences contribute 3-4× more to gradients than shorter ones, creating high variance in gradient updates (larger seq lens are preferred over shorter ones). Based on my initial tests, per-token loss results in much more stable training with acceptable loss values and consistent gradient norms. I plan to conduct a few ablation experiments comparing both approaches to show that per-token loss is indeed a better approach for this task.

### Workarounds for vLLM Model–Based Intermediate Evaluation
If you follow the steps in [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_alignment.pdf) to run intermediate evaluation, you will run into errors related to vLLM initialization and weight loading. Here’s what I had to change.

#### Use the `colocate` Approach for vLLM Initialization
- The separate-GPU inference setup described in the assignment no longer works with vLLM 0.7.x and above. The initialization logic has changed, and the assignment’s code is outdated.
*-The simpler solution is to use the `colocate` initialization strategy, where you create the vLLM model on the same device as the SFT policy model.
- Make sure the initialization code runs inside a `main` function so that vLLM’s multiprocessing can launch correctly. Also set appropriate values for `gpu_memory_utilization`, `max_model_len`, and `max_num_seqs`.

#### Fixing the `LLMEngine` Missing `model_executor` Attribute
*-Loading updated model weights into the vLLM instance caused an `LLMEngine` error:
  `AttributeError: 'LLMEngine' object has no attribute 'model_executor'`.
- Two solutions work:
  - Downgrade vLLM to **0.10.2**, or
  - If using **vLLM 0.11.0**, set the environment variable `VLLM_ENABLE_V1_MULTIPROCESSING=0` at the start of the script.

#### Fixing `ValueError: There is no module or parameter named '_orig_mod' in Qwen2ForCausalLM`
- This happens when using a compiled model in `sft_microbatch_train_step`.
- The issue comes from the compiled model’s original weights being referenced under the `_orig_mod` attribute. Make sure to load the policy weights from `_orig_mod` attribute into the vLLM instance in `load_policy_into_vllm_instance`.

Please refer to my following chatGPT [conversation](https://chatgpt.com/share/691d6824-4564-8012-a2c9-587c599c45ec) for more details on the workarounds

## SFT Training Results
Below are the evaluation results across different SFT training runs:

| Run Name | Reward Accuracy | Format Accuracy |
| :------- | :-------------- | :-------------- |
| baseline | 0.0288 | 0.1438 |
| run_all | 0.4214 | 0.9924 |
| run_filtered | 0.5204 | 0.9906 |
| run_filtered-res-len | 0.5106 | 0.9898 |
| run_filtered-2epoch | 0.5336 | 0.9926 |

**Notes**:
- Different runs:
  - All runs except `run_filtered-2epoch` are trained for the same number of steps.
  - **baseline**: Represents the untrained Qwen2.5-Math-1.5B model accuracy.
  - **run_all**: Training data is full ~4.8K examples.
  - **run_filtered**: Training on filtered 4.8K dataset (~3.6K examples) where we removed the reasoning traces that lead to wrong answers.
  - **run_filtered-res-len**: Uses non-response length normalized loss similar to what's mentioned in the assignment (i.e., no per-token loss).
  - **run_filtered-2epoch**: Same run as `run_filtered` but trained for 2 epochs.
- All training runs can be found in the [wandb's sft workspace](https://wandb.ai/garg-aayush/sft). The training was performed on an RTX 6000 GPU (96GB) rented on RunPod (CUDA 12.8, PyTorch 2.8, vLLM 0.11.0).
- Validation set: 
  - The evaluation results shown in the table above are calculated on the full validation dataset of ~5K examples (same as provided in the assignment). 
  - However during training, the W&B logs show evaluation metrics calculated on a randomly sampled subset of 1k examples from the validation set. This subset was sampled at the start of each training run to monitor training progress more efficiently without running full evaluation at each checkpoint.

## Optional Assignment 5
- [Optional Supplementary Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

### Run baseline evaluations on `Llama-3.1-8B` model
| Eval | Accuracy |
| :--- | :------- |
| MMLU | 0.58 |
| GSM8K | 0.16 |
| AlpacaEval | 1.49% |
| Simple Safety Tests | 0.62 |

### Running alpaca_eval

- As per the assignment, I'm using [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) to evaluate the quality of my model's dialogue responses through an LLM-as-a-judge approach. AlpacaEval is a benchmark that measures how well language models respond to open-ended instructions by having a strong judge model compare outputs pairwise. Here, I compare the base `Llama-3.1-8B` (or finetuned's model responses against GPT-4 baseline outputs across 805 diverse instructions given in the `alpaca_eval.jsonl` file. *However, unlike the assignment, here I use a judge model (`Llama-3.3-70B-Instruct`) via Fireworks API.*
- The script: `evaluate_dialogue_alpaca_eval.py` orchestrates the entire process. 
  - It first uses vLLM to generate responses for all instructions from `alpaca_eval.jsonl` and save it to `baseline_alpaca_eval_outputs.json` in the results directory. 
  - After generation, the script automatically calls the alpaca_eval library, which loads the judge configuration from `configs/configs.yaml` (specifying the Llama-3.3-70B-Instruct judge model), compares my outputs against the GPT-4 reference outputs from `alpaca_eval_gpt4_baseline.json`, and produces annotations and a leaderboard CSV with the final win rate.
- To run this evaluation, make sure to set the `FIREWORKS_API_KEY` environment variable since the judge model runs through Fireworks API (the script automatically maps this to `OPENAI_API_KEY` for compatibility with alpaca_eval). 

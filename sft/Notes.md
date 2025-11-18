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
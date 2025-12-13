import os

# disable v1 multiprocessing to avoid 'LLMEngine' object has no attribute 'model_executor' error in vLLM 0.11.0
# otherwise downgrade vllm to 0.10.2
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
import inspect
import json
import random
import time
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from utils.dataloader import SftTrainDataLoaderLite
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.helper_fns import (evaluate_vllm, init_vllm,
                              load_policy_into_vllm_instance, prepare_val_data,
                              pretty_print)
from utils.post_train import (get_response_log_probs, sft_eval_step,
                              sft_microbatch_train_step,
                              tokenize_prompt_and_output)
from vllm import LLM, SamplingParams

# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
# wandb tracking setup
seed = 1337
wandb_project = "expert-iter"
wandb_run_name = "test"

# Model config
model_name = "Qwen/Qwen2.5-Math-1.5B"
dtype = "bfloat16"  # "float16" or "bfloat16"
attention_type = "flash_attention_2"
use_compile = True

# Device & vLLM config
device = "cuda:0"  # please use a GPU for training
device_type = "cuda" if device.startswith("cuda") else "cpu"

# train and val ta
train_data_file = "/home/aayush/DATA/EXPERT-ITER/sft_gpt-oss-120b_filtered.jsonl"
prompt_template_file = "/home/aayush/DATA/EXPERT-ITER/r1_zero.prompt"
val_data_file = "/home/aayush/DATA/EXPERT-ITER/val.jsonl"

# Training hyperparameters
total_batch_size = 8
micro_batch_size = 2
max_grad_acc_steps = total_batch_size // micro_batch_size
val_batch_size = 4
learning_rate = 1e-5 
max_steps = 38 # ~1 epoch for 4836 examples (non-filtered), ~28 steps for 3496 examples (filtered)
grad_norm_clip = 1.0
use_per_token_loss = True # use per-token loss instead of per-sequence loss
normalize_constant = 1.0 # normalization constant for the loss
batch_per_ei = 512
num_ei = 5
num_rollouts = 4 # number of outputs to generate for each example

# Checkpointing & logging
output_dir = f"/home/aayush/RESULTS/{wandb_run_name}"
checkpoint_interval = 10
run_intermediate_eval = True # run intermediate evaluation on the validation set
log_eval_examples_to_jsonl = True
max_val_examples = 1000  # maximum number of validation examples to evaluate on
num_val_examples_to_log = 20 # number of validation examples to log to wandb
eval_interval = 10  # evaluate on val dataset

# vLLM config
vllm_dtype = "bfloat16"  # vLLM expects string, not torch dtype
vllm_gpu_memory_utilization = 0.2  # reserve 20% of the GPU memory
vllm_max_model_len = 2048  # maximum model length
vllm_max_num_seqs = 128  # maximum number of sequences
vllm_sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "min_tokens": 4,
    "max_tokens": 1024,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
    "n": num_rollouts
}


# -------------------------------------------------------------#
# Helper functions
# -------------------------------------------------------------#
def create_ei_filtered_data(prompt_template_file: str, data_file: str, max_examples: int=512, 
                            vllm_model: LLM=None, vllm_sampling_params_obj: SamplingParams=None,
                            reward_fn: Callable[[str, str], dict[str, float]]=None, filtered_data_file: str=None):
    # sample a batch of max_examples examples from the data
    prompts, examples = sample_batch(prompt_template_file, data_file, max_examples)
    print(f"Sampled {len(prompts)} examples from the data")
    pretty_print(examples[0], title="Example example")
    pretty_print(prompts[0], title="Example prompt")
    
    # filter the examples using the vLLM model
    filtered_train_results, filtered_acc_dict = filter_data(vllm_model, reward_fn, prompts, examples, vllm_sampling_params_obj)
    pretty_print(filtered_train_results[0], title="Example filtered train result")
    print(f"Filtered {len(filtered_train_results)} examples out of {len(examples)} examples")
    print(f"Accuracy: {filtered_acc_dict['avg_acc']:.4f}, Format accuracy: {filtered_acc_dict['avg_format_acc']:.4f}")
    
    # save the filtered data to a jsonl file list of dicts
    with open(filtered_data_file, "w") as f:
        json.dump(filtered_train_results, f, indent=2)
    return filtered_data_file

def sample_batch(prompt_template_file: str, data_file: str, max_examples: int=512):
    # read the data file and prompt template file
    with open(prompt_template_file, "r") as f:
        prompt_template = f.read()
    with open(data_file, "r") as f:
        data = json.load(f)

    # sample the validation data
    total_examples = len(data)
    if max_examples > total_examples:
        print(f"Warning: max_examples={max_examples} is greater than the total_examples={total_examples}, using all {total_examples} examples")
    else:
        data = random.sample(data, max_examples)
    
    # create the list of prompts and baseline results dict
    prompts, examples = zip(*[
        (prompt_template.format(question=data[i]["problem"]), 
        {
            "problem": data[i]["problem"], 
            "reasoning_trace": data[i]["reasoning_trace"],
            "expected_answer": data[i]["expected_answer"],
            "extracted_answer": data[i]["extracted_answer"]
        }
        )
        for i in range(len(data))
    ])
    return list(prompts), list(examples)

def filter_data(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    results: List[dict],
    sampling_params: SamplingParams
) -> None:
    
    # generate the prompts
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # calculate the reward using the reward function
    # and add the outputs to the baseline results
    acc_dict = {"avg_acc": 0.0,
                "avg_format_acc": 0.0,
                }
    filtered_results = []
    for i, (output_list, result) in enumerate(zip(outputs, results)):
        for j, output in enumerate(output_list.outputs):
            output_text = output.text
            reward = reward_fn(output_text, str(result["expected_answer"]).strip())
            if reward["reward"] > 0.0:
                filtered_results.append(result)
                acc_dict["avg_acc"] += reward["reward"]
                acc_dict["avg_format_acc"] += reward["format_reward"]
                break
    total_examples = len(results)
    total_filtered_examples = len(filtered_results)
    for key in acc_dict.keys():
        acc_dict[key] /= total_examples
    
    return filtered_results, acc_dict

# -------------------------------------------------------------#
# Seed and precision setup
# -------------------------------------------------------------#
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# use tf32
torch.set_float32_matmul_precision("high")


# -------------------------------------------------------------#
# Main function
# -------------------------------------------------------------#
# wrap the code in a main function to allow for vllm initialization that uses multi-processing
if __name__ == '__main__':
    # print config
    input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
    pretty_print(input_config, title="Input config")
    
    # -------------------------------------------------------------#
    # Assertions and other setup
    # -------------------------------------------------------------#
    # assertions to ensure the training can be run
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please use a GPU for training")
    if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype = "float16"
        vllm_dtype = "float16"
        raise ValueError("bfloat16 is not supported on this device, please use a different dtype")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


    # -------------------------------------------------------------#
    # Initialize the vLLM model for inference 
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the vLLM model for inference...", title="vLLM model initialization")
    vllm_init_params = {
        "model": model_name,
        "gpu_memory_utilization": vllm_gpu_memory_utilization,
        "max_model_len": vllm_max_model_len,
        "max_num_seqs": vllm_max_num_seqs
    }
    pretty_print(vllm_init_params, title="vLLM model initialization parameters")
    vllm_model = init_vllm(seed, vllm_init_params)
    # create sampling parameters object
    vllm_sampling_params_obj = SamplingParams(**vllm_sampling_params)
    
    # -------------------------------------------------------------#
    # Initialize the tokenizer and model
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the tokenizer and model...", title="Tokenizer and model initialization")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attention_type, device_map=device)
    # compile the model
    if use_compile:
        model = torch.compile(model)
    
    
    # -------------------------------------------------------------#
    # Setup the optimizer
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the optimizer...", title="Optimizer initialization")
    # check if fused AdamW is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    print(f"Using fused AdamW: {use_fused}")
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=use_fused)
    print(optimizer)
    
    # -------------------------------------------------------------#
    # Training loop
    # -------------------------------------------------------------#
    pretty_print(f"Starting the training loop...", title="Training loop")
    model.train()
    
    # outer loop: expert iteration
    for ei in range(num_ei):
        torch.cuda.empty_cache()
        if ei > 0:
            print(f"Loading the model weights from the previous expert iteration to vLLM model...")
            load_policy_into_vllm_instance(model, vllm_model)
        
        # sample a batch of batch_per_ei examples from the data and filter them and save the filtered data to a jsonl file
        pretty_print(f"Sampling a batch of {batch_per_ei} examples from the data and filtering them for expert iteration {ei}...", title="Filtering train data")
        cur_train_data_file = create_ei_filtered_data(prompt_template_file, train_data_file, batch_per_ei, 
                                                           vllm_model, vllm_sampling_params_obj, r1_zero_reward_fn, 
                                                           f"./tmp_{ei}.json")
        print(f"Filtered data saved to {cur_train_data_file}")
        torch.cuda.empty_cache()
        
        # Setup the data loader
        pretty_print(f"Initializing the data loader...", title="Data loader initialization")
        # initialize the data loader
        train_data_loader = SftTrainDataLoaderLite(cur_train_data_file, prompt_template_file, micro_batch_size, seed + ei)
        # print the first batch of data
        pretty_print("", title="First batch of data")
        for i, item in enumerate(train_data_loader.get_batch()):
            pretty_print(item, title=f"Example data {i}")
        # reset the data loader pointer
        train_data_loader.reset(only_ptr=True)
        print(f"Number of batches in current expert iteration: {len(train_data_loader)}")

        cur_grad_acc_steps = min(max_grad_acc_steps, len(train_data_loader) // micro_batch_size)
        max_steps = max(len(train_data_loader) // total_batch_size, 1) 
        print(f"Maximum number of steps: {max_steps}\nCurrent gradient accumulation steps: {cur_grad_acc_steps}")
        
        
        for step in range(max_steps):
            # start time    
            t0 = time.time()
            
            # set the optimizer to zero grad
            optimizer.zero_grad(set_to_none=True)
            
            loss_accum = 0.0
            entropy_accum = 0.0
            total_response_tokens = 0
            for micro_step in range(cur_grad_acc_steps):
                # get a batch of data
                batch = train_data_loader.get_batch()
                # tokenize the batch
                tokenized_batch = tokenize_prompt_and_output(prompt_strs=[item["prompt"] for item in batch],
                                                            output_strs=[item["response"] for item in batch],
                                                            tokenizer=tokenizer)
                input_ids, labels, response_mask = tokenized_batch["input_ids"].to(device), tokenized_batch["labels"].to(device), tokenized_batch["response_mask"].to(device)
                
                # calculate the loss
                with torch.autocast(device_type=device_type, dtype=dtype):
                    # get the log-probs of the response given the prompt and the token entropy
                    # log_probs: (batch_size, sequence_length), token_entropy: (batch_size, sequence_length)
                    response_log_probs = get_response_log_probs(model, input_ids, labels, True)
                    # calculate the loss
                    loss, loss_metrics = sft_microbatch_train_step(response_log_probs["log_probs"], response_mask, cur_grad_acc_steps, 
                                                                normalize_constant=normalize_constant, per_token_loss=use_per_token_loss)
                    # accumulate the loss
                    loss_accum += loss.detach()
                
                # accumulate the token entropy (only for response tokens)
                # move it outside the autocast to save memory overhead (minor)
                entropy_accum += (response_log_probs["token_entropy"] * response_mask).sum().detach()
                total_response_tokens += response_mask.sum().detach()
                
            # global norm gradient clipping at 1.0
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            # update the model parameters
            optimizer.step()
        
        
            # calculate average token entropy across all response tokens
            avg_token_entropy = (entropy_accum / total_response_tokens).item()
            # print the loss metrics
            dt = time.time() - t0
            print(f"step: {step:0d} | loss: {loss_accum.item():.4f} | avg_entropy: {avg_token_entropy:.4f} | lr: {learning_rate:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}s")
            
           
            torch.cuda.empty_cache()

# # Save the evaluation examples table
# if run_intermediate_eval:
#     wandb.log({"eval/examples": val_examples_table})
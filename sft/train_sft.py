import os

# disable v1 multiprocessing to avoid 'LLMEngine' object has no attribute 'model_executor' error in vLLM 0.11.0
# otherwise downgrade vllm to 0.10.2
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
import inspect
import json
import random
import time
from typing import List

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
from vllm import SamplingParams

# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
# wandb tracking setup
seed = 1337
wandb_project = "sft"
wandb_run_name = "run_filtered"

# Model config
model_name = "/root/qwen"
dtype = "bfloat16"  # "float16" or "bfloat16"
attention_type = "flash_attention_2"
use_compile = True

# Device & vLLM config
device = "cuda:0"  # please use a GPU for training
device_type = "cuda" if device.startswith("cuda") else "cpu"

# train and val ta
train_data_file = "/root/data/sft_gpt-oss-120b_filtered.jsonl"
prompt_template_file = "/root/data/r1_zero.prompt"
val_data_file = "/root/data/val.jsonl"

# Training hyperparameters
total_batch_size = 128
micro_batch_size = 2
grad_acc_steps = total_batch_size // micro_batch_size
val_batch_size = 4
learning_rate = 1e-4
max_steps = 38 # ~1 epoch for 4836 examples (non-filtered), ~28 steps for 3496 examples (filtered)
grad_norm_clip = 1.0
use_per_token_loss = True # use per-token loss instead of per-sequence loss
normalize_constant = 1.0 # normalization constant for the loss

# Checkpointing & logging
output_dir = f"/root/results/{wandb_run_name}"
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
    "max_tokens": 1024,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True
}


# -------------------------------------------------------------#
# Helper functions
# -------------------------------------------------------------#
def log_eval_generations(prompts: List[str], results: List[dict], indices_to_log: List[int], step: int, examples_table: wandb.Table, eval_examples_dir: str=None) -> None:
    """Log the generations for the evaluation prompts."""
    
    if eval_examples_dir is not None:
        eval_examples = []
    
    # update the example table
    for i in indices_to_log:
        cur_result = results["results"][i]
        examples_table.add_data(
            i, step, prompts[i], 
            cur_result["expected_answer"], cur_result["output"], 
            cur_result["reward"]["reward"], cur_result["reward"]["format_reward"], cur_result["reward"]["answer_reward"]
            )
        
        # log the evaluation examples to jsonl
        if eval_examples_dir is not None:
            eval_examples.append({"index": i, "input_prompt": prompts[i], **cur_result})
    
    # log the evaluation examples to jsonl
    if eval_examples_dir is not None:
        with open(os.path.join(eval_examples_dir, f"eval_examples_step_{step}.jsonl"), "w") as f:
            json.dump(eval_examples, f, indent=2)

@torch.no_grad()
def evaluate_val_data(model, val_prompts, results, val_batch_size):
    """Evaluate the model on the validation data and return the avg_loss, avg_token_entropy, avg_correct_response_length, avg_incorrect_response_length"""
    
    model.eval()
    
    # track different metrics
    loss_accum = 0.0
    token_entropy_accum = 0.0
    correct_response_tokens = 0
    incorrect_response_tokens = 0
    num_batches = len(val_prompts) // val_batch_size
    num_correct = 0
    num_incorrect = 0
    
    for i in range(0, len(val_prompts), val_batch_size):
        # get the batch of prompts, results, and reward flags
        batch_prompts = val_prompts[i:i+val_batch_size]
        batch_results = [res["output"] for res in results[i:i+val_batch_size]]
        batch_reward_flags = [res["reward"]["reward"] > 0 for res in results[i:i+val_batch_size]]
        
        # tokenize the batch
        tokenized_batch = tokenize_prompt_and_output(prompt_strs=batch_prompts,
                                                        output_strs=batch_results,
                                                        tokenizer=tokenizer)
        input_ids, labels, response_mask = tokenized_batch["input_ids"].to(device), tokenized_batch["labels"].to(device), tokenized_batch["response_mask"].to(device)
        
        # calculate the loss and token entropy
        with torch.autocast(device_type=device_type, dtype=dtype):
            # log_probs: (batch_size, sequence_length), token_entropy: (batch_size, sequence_length)
            response_log_probs = get_response_log_probs(model, input_ids, labels, True)
            # calculate the loss
            loss = sft_eval_step(response_log_probs["log_probs"], response_mask, normalize_constant, use_per_token_loss)
        # accumulate the loss and token entropy
        loss_accum += loss.detach()
        token_entropy_accum += (response_log_probs["token_entropy"] * response_mask).sum().detach()
        
        # split the response token counts into correct and incorrect
        for j, corr_flag in enumerate(batch_reward_flags):
            if corr_flag:
                correct_response_tokens += response_mask[j].sum().detach()
                num_correct += 1
            else:
                incorrect_response_tokens += response_mask[j].sum().detach()
                num_incorrect += 1
        
    # calculate the average loss, token entropy, and response length
    avg_loss = loss_accum / num_batches
    avg_token_entropy = token_entropy_accum / (correct_response_tokens + incorrect_response_tokens)
    avg_correct_response_length = correct_response_tokens / num_correct if num_correct > 0 else 0.0
    avg_incorrect_response_length = incorrect_response_tokens / num_incorrect if num_incorrect > 0 else 0.0
    
    model.train()
    
    return avg_loss.item(), avg_token_entropy.item(), avg_correct_response_length.item(), avg_incorrect_response_length.item()
    
    
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
    # Initialize wandb
    # -------------------------------------------------------------#
    pretty_print(f"Initializing wandb project {wandb_project} and run {wandb_run_name}...", title="Wandb initialization")
    wandb.init(project=wandb_project, name=wandb_run_name, config=input_config)

    # x/y-axis for train and eval steps
    wandb.define_metric("step_train")
    wandb.define_metric("step_eval")

    # metrics that start with train/ are tied to train_step and vice versa for eval/
    wandb.define_metric("train/*", step_metric="step_train")
    wandb.define_metric("eval/*", step_metric="step_eval")


    # -------------------------------------------------------------#
    # Prepare the validation data
    # -------------------------------------------------------------#
    if run_intermediate_eval:
        pretty_print(f"Preparing the validation data of {max_val_examples} examples from {val_data_file}...", title="Validation data preparation")
        val_prompts, val_baseline_results = prepare_val_data(val_data_file, prompt_template_file, max_val_examples)
        # get random indices to log
        val_indices_to_log = random.sample(range(len(val_prompts)), num_val_examples_to_log)
        # I create the table here so it only contains data for the current evaluation step
        val_examples_table = wandb.Table(columns=["val_index", "eval_step", "input_prompt", "expected_answer", "generated_output", "reward", "format_reward", "answer_reward"])
        # print the validation prompts and baseline results
        pretty_print(val_prompts[:3], title="Example validation prompt")
        pretty_print(val_baseline_results[5], title="Example validation baseline result")
        
        if log_eval_examples_to_jsonl:
            # create the jsonl directory in the output directory
            eval_examples_dir = os.path.join(output_dir, "eval_examples")
            os.makedirs(eval_examples_dir, exist_ok=True)
        else:
            eval_examples_dir = None


    # -------------------------------------------------------------#
    # Initialize the vLLM model for inference 
    # -------------------------------------------------------------#
    if run_intermediate_eval:
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
    # Setup the data loader
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the data loader...", title="Data loader initialization")
    # initialize the data loader
    train_data_loader = SftTrainDataLoaderLite(train_data_file, prompt_template_file, micro_batch_size, seed)
    # print the first batch of data
    pretty_print("", title="First batch of data")
    for i, item in enumerate(train_data_loader.get_batch()):
        pretty_print(item, title=f"Example data {i}")
    # reset the data loader pointer
    train_data_loader.reset(only_ptr=True)
    

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

    # training loop
    for step in range(1, max_steps+1):
        # start time
        t0 = time.time()
        
        # set the optimizer to zero grad
        optimizer.zero_grad(set_to_none=True)
        
        # evaluate the model on the validation data, track the eval time
        if run_intermediate_eval and (step == 1 or step % eval_interval == 0 or step == max_steps):
            t1 = time.time()
            if step != 0:
                # load the model weights from the checkpoint
                print(f"Loading the model weights from the current checkpoint to vLLM model...")
                load_policy_into_vllm_instance(model, vllm_model)
            
            # generate the responses using the vLLM model
            vllm_eval_results = evaluate_vllm(vllm_model, r1_zero_reward_fn, val_prompts, val_baseline_results, vllm_sampling_params_obj)
            
            # clear torch cache (to save memory)
            torch.cuda.empty_cache()
            
            # log the evaluation generations
            log_eval_generations(val_prompts, vllm_eval_results, val_indices_to_log, step, val_examples_table, eval_examples_dir)
            
            # evaluate the model on the validation data and track the avg_loss, avg_token_entropy, avg_response_length (correct responses and incorrect responses)
            eval_avg_loss, eval_avg_token_entropy, avg_correct_res_len, avg_incorrect_res_len = evaluate_val_data(model, val_prompts, vllm_eval_results["results"], val_batch_size)
            
            # eval metrics
            dt = time.time() - t1
            eval_avg_format_acc = vllm_eval_results['accuracy']['avg_format_acc']
            eval_avg_acc = vllm_eval_results['accuracy']['avg_acc']
            # log the evaluation metrics to wandb
            wandb.log({
                "step_eval": step,
                "eval/avg_format_acc": eval_avg_format_acc,
                "eval/avg_acc": eval_avg_acc,
                "eval/loss": eval_avg_loss,
                "eval/avg_token_entropy": eval_avg_token_entropy,
                "eval/avg_correct_response_length": avg_correct_res_len,
                "eval/avg_incorrect_response_length": avg_incorrect_res_len,
                "eval/dt": dt
            })
            # print the evaluation metrics
            print(f"eval_step: {step:0d} | loss: {eval_avg_loss:.4f} | avg_entropy: {eval_avg_token_entropy:.4f} | avg_correct_res_len: {avg_correct_res_len:.2f} | avg_incorrect_res_len: {avg_incorrect_res_len:.2f} | avg_format_acc: {eval_avg_format_acc:.4f} | avg_acc: {eval_avg_acc:.4f} | dt: {dt:.2f}s")
            
            # clear torch cache (to save memory)
            torch.cuda.empty_cache()
        
        loss_accum = 0.0
        entropy_accum = 0.0
        total_response_tokens = 0
        for micro_step in range(grad_acc_steps):
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
                loss, loss_metrics = sft_microbatch_train_step(response_log_probs["log_probs"], response_mask, grad_acc_steps, 
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
        
        # log the training metrics to wandb
        train_log_dict = {
            "step_train": step,
            "train/loss": loss_accum.item(),
            "train/avg_token_entropy": avg_token_entropy,
            "train/lr": learning_rate,
            "train/norm": norm.item(),
            "train/dt": dt
        }
        wandb.log(train_log_dict)
        
        if (step % checkpoint_interval == 0) or (step == max_steps) or (step == max_steps):
            model.save_pretrained(output_dir + f"/checkpoint_{step}")
            tokenizer.save_pretrained(output_dir + f"/checkpoint_{step}")
        
        # clear cache
        # Not an ideal solution as it creates overhead for release and re-allocation of memory, we lose the benefit of memory caching.
        # However, I am using it to avoid OOMs error at same steps due to variable memory usage (variable sequence length per batch), large memory spikes due to gradient accumulation and unfreed memory (memory leaks).
        if step % eval_interval == 0: torch.cuda.empty_cache()
        

# Save the evaluation examples table
if run_intermediate_eval:
    wandb.log({"eval/examples": val_examples_table})
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
from utils.helper_fns import pretty_print
from utils.post_train import (get_response_log_probs,
                              sft_microbatch_train_step,
                              tokenize_prompt_and_output)

# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
seed = 1337
wandb_project = "sft"
wandb_run_name = "test"
# model
model_name = "Qwen/Qwen2.5-Math-1.5B"
dtype = "bfloat16" # "float16" or "bfloat16"
attention_type = "flash_attention_2"
device = "cuda:0" # please use a GPU for training
use_compile = True
use_per_token_loss = False
normalize_constant = 1.0
# training data
train_data_file = "data/sft_gpt-oss-120b_filtered.jsonl"
prompt_template_file = "data/r1_zero.prompt"
total_batch_size = 128
micro_batch_size = 2
learning_rate = 1e-4
max_steps = 40
grad_norm_clip = 1.0
output_dir = "results/filtered"
grad_acc_steps = total_batch_size // micro_batch_size
device_type = "cuda" if device.startswith("cuda") else "cpu"
checkpoint_interval = 10


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
    raise ValueError("bfloat16 is not supported on this device, please use a different dtype")
dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

# set the seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# use tf32
torch.set_float32_matmul_precision("high")

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
# # Initialize the tokenizer and model
# # -------------------------------------------------------------#
pretty_print(f"Initializing the tokenizer and model...", title="Tokenizer and model initialization")
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attention_type, device_map=device)
# compile the model
if use_compile:
    model = torch.compile(model)

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
for step in range(max_steps):
    
    t0 = time.time()
    # set the optimizer to zero grad
    optimizer.zero_grad(set_to_none=True)
    
    # accumulate gradients over multiple steps
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
    # clear cache
    # Not an ideal solution as it creates overhead for release and re-allocation of memory, we lose the benefit of memory caching.
    # However, I am using it to avoid OOMs error at same steps due to variable memory usage (variable sequence length per batch), large memory spikes due to gradient accumulation and unfreed memory (memory leaks).
    torch.cuda.empty_cache()
    
    
    # calculate average token entropy across all response tokens
    avg_token_entropy = (entropy_accum / total_response_tokens).item()
    # print the loss metrics
    dt = time.time() - t0
    print(f"step: {step:0d} | loss: {loss_accum.item():.4f} | avg_entropy: {avg_token_entropy:.4f} | avg_len: {loss_metrics["avg_response_length"]:.2f} | lr: {learning_rate:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}s")
    
    # log the training metrics to wandb
    train_log_dict = {
        "step_train": step,
        "train/loss": loss_accum.item(),
        "train/avg_token_entropy": avg_token_entropy,
        "train/avg_response_length": loss_metrics["avg_response_length"],
        "train/lr": learning_rate,
        "train/norm": norm.item(),
        "train/dt": dt
    }
    wandb.log(train_log_dict)
    
    if ((step + 1) % checkpoint_interval == 0) or (step + 1 == max_steps):
        model.save_pretrained(output_dir + f"/checkpoint_{step+1}")
        tokenizer.save_pretrained(output_dir + f"/checkpoint_{step+1}")
import inspect
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.helper_fns import pretty_print
from utils.instruct_dataset import InstructFinetuneDataset, iterate_dataset

# Disable tokenizer parallelism to avoid conflicts with DataLoader multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
# wandb tracking setup
seed = 1337
wandb_project = "sft_instruct"
wandb_run_name = "test"

# Model config
# model_name = "meta-llama/Llama-3.1-8B"
model_name = "Qwen/Qwen2.5-Math-1.5B"

dtype = "bfloat16"  # "float16" or "bfloat16"
attention_type = "flash_attention_2"
use_compile = False

# Device
device = "cuda"  # "cuda" or "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"

# train data params
train_data_file = "/home/aayush/DATA/SFT/train.jsonl"
prompt_template_file = "/home/aayush/repos/building-from-scratch/sft/data/alpaca_sft.prompt"
train_num_workers = 4

# eval data params
eval_batch_size = 4
eval_num_workers = 0
eval_data_file = "/home/aayush/DATA/SFT/test.jsonl"
eval_interval = 50

# common data params
seq_length = 512
apply_masking = False

# Training hyperparameters
total_batch_size = 32
micro_batch_size = 2
grad_acc_steps = total_batch_size // micro_batch_size
max_lr = 2e-5
min_lr = max_lr * 0.1
max_steps = -100
num_epochs = 1
grad_norm_clip = 1.0
warmup_ratio = 0.03
cache_clear_interval = 10

# Checkpointing & logging
output_dir = f"/home/aayush/RESULTS/SFT/instruct/{wandb_run_name}"
checkpoint_interval = 100

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
# Assertions and other setup
# -------------------------------------------------------------#
# assertions to ensure the training can be run
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available, please use a GPU for training")
if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
    dtype = "float16"
dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
assert total_batch_size % micro_batch_size == 0, f"Total batch size: {total_batch_size} must be divisible by micro batch size: {micro_batch_size}"
for p in [train_data_file, eval_data_file]:
    assert os.path.exists(p), f"File {p} does not exist"

# print config
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")
    
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
# Setup the dataloaders
# -------------------------------------------------------------#
pretty_print(f"Setup up the dataloaders...", title="Dataloaders setup")
print(f"Setting up the training dataloader...")
# initialize the training dataloader
train_dataset = InstructFinetuneDataset(
    tokenizer=tokenizer,
    dataset_path=train_data_file, 
    seq_length=seq_length,
    shuffle=True,
    alpaca_prompt_template_file=prompt_template_file,
    apply_masking=apply_masking)
train_dataloader = iterate_dataset(train_dataset, micro_batch_size, shuffle=True, num_workers=train_num_workers, pin_memory=True)
print(f"Length of training dataloader: {len(train_dataloader)}")


# initialize the evaluation dataloader
print(f"Setting up the evaluation dataloader...")
eval_dataset = InstructFinetuneDataset(
    tokenizer=tokenizer,
    dataset_path=eval_data_file, 
    seq_length=seq_length,
    shuffle=False,
    alpaca_prompt_template_file=prompt_template_file,
    apply_masking=apply_masking)
eval_dataloader = iterate_dataset(eval_dataset, eval_batch_size, shuffle=False, num_workers=eval_num_workers)
print(f"Length of evaluation dataloader: {len(eval_dataloader)}")

# -------------------------------------------------------------#
# Setup the optimizer
# -------------------------------------------------------------#
# set the max_steps if it is not set
if max_steps <= 0:
    max_steps = num_epochs * math.ceil(len(train_dataloader) / grad_acc_steps)
warmup_steps = int(max_steps * warmup_ratio)
print(f"Warmup steps: {warmup_steps}")
print(f"Max. steps: {max_steps}")

# cosine decay learning-rate scheduler with linearwarmup
def get_lr(step):
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if step > max_steps, return min_lr
    if step > max_steps:
        return min_lr
    # 3) otherwise, use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



pretty_print(f"Initializing the optimizer...", title="Optimizer initialization")
# check if fused AdamW is available
fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == "cuda"
print(f"Using fused AdamW: {use_fused}")
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=use_fused)
print(optimizer)

# -------------------------------------------------------------#
# Training loop
# -------------------------------------------------------------#
pretty_print(f"Starting the training loop...", title="Training loop")
@torch.no_grad()
def evaluate_val_data(model, eval_dataloader):
    """Evaluate the model on the validation data"""
    model.eval()
    start_time = time.time()
    eval_loss = 0.0
    for batch in eval_dataloader:
        input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        with torch.autocast(device_type=device_type, dtype=dtype):
            logits = model(input_ids).logits
            logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
            eval_loss += loss.detach()
    eval_loss = eval_loss / len(eval_dataloader)
    model.train()
    torch.cuda.empty_cache()
    return {"eval_loss": eval_loss.item(), "eval_time": time.time() - start_time}

vocab_size = model.config.vocab_size
print(f"vocab size: {vocab_size}")
model.train()
train_iterator = iter(train_dataloader)

for step in range(max_steps):
    t0 = time.time()
    # evaluate the model on the validation data
    if step % eval_interval == 0 or step == max_steps - 1:
        eval_dict = evaluate_val_data(model, eval_dataloader)
        print(f"eval_step: {step:04d} | eval_loss: {eval_dict['eval_loss']:.4f} | dt: {eval_dict['eval_time']:.2f}s")
        wandb.log({"step_eval": step, "eval/loss": eval_dict['eval_loss'], "eval/dt": eval_dict['eval_time']})
    # set the optimizer to zero grad
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(grad_acc_steps):
        # get a batch of data
        try:
            batch = next(train_iterator)
        # if the iterator is exhausted, reset it and get the next batch
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        # forward pass in autocast
        with torch.autocast(device_type=device_type, dtype=dtype):
            # # this is same as the below used code
            # outputs = model(input_ids, labels=labels)
            # loss = outputs.loss
            # # calculate the loss
            # shift for the next token prediction
            # reshape: logits (b, sl, v) -> (b*sl, v) and labels (b, sl) -> (b*sl)
            # .contiguous() is needed to ensure the memory is contiguous
            logits = model(input_ids).logits
            logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
        # scale the loss by the gradient accumulation steps
        loss = loss / grad_acc_steps
        loss_accum += loss.detach()
        # backward pass
        loss.backward()
    # global norm gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
    # update the model parameters
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    
    # log the training loss
    dt = time.time() - t0
    print(f"step: {step:04d} | train_loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}s")
    wandb.log({"step_train": step, "train/loss": loss_accum.item(), "train/lr": lr, "train/norm": norm.item(), "train/dt": dt})
    
    # save the model and tokenizer
    if (step > 0 and step % checkpoint_interval == 0) or step == max_steps - 1:
            ckpt_dir = output_dir + f"/checkpoint_{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
    
    # clear cache
    if step % cache_clear_interval == 0: torch.cuda.empty_cache()
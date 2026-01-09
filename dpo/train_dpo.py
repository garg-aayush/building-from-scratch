import math
import os
import random
import time

import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dpo_dataset import DpoFinetuneDataset
from utils.dpo_loss import dpo_pref_accuracy, per_instance_dpo_loss
from utils.helper_fns import pretty_print

# Disable tokenizer parallelism to avoid conflicts with DataLoader multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
# wandb tracking setup
seed = 1337
wandb_project = "dpo"
wandb_run_name = "test_run"

# Model config
model_name = "/home/models/llama31-8b"

dtype = "bfloat16"  # "float16" or "bfloat16"
attention_type = "flash_attention_2"
use_compile = False # for CUDA 2.6, H100 false otherwise OOMs error, RTX6000 CUDA 2.8 true 

# Device
device_policy = "cuda:0"
device_ref = "cuda:1"
device_type = "cuda" if device_policy.startswith("cuda") else "cpu"

# Data params
data_file = "data/examples.jsonl"
prompt_template_file = "data/alpaca_sft.prompt"
num_val = 500  # 1% of ~50k examples

# Training hyperparameters
total_batch_size = 64
micro_batch_size = 1  # DPO processes one example at a time in the current loss implementation
grad_acc_steps = total_batch_size // micro_batch_size
max_lr = 1e-6  # DPO typically uses lower learning rates
max_steps = -1
num_epochs = 1
grad_norm_clip = 1.0
warmup_ratio = 0.1
cache_clear_interval = 100

# DPO-specific hyperparameters
beta = 0.1  # KL penalty coefficient
# Eval params
eval_interval = 100
# Checkpointing & logging
output_dir = f"results/dpo/{wandb_run_name}"

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
if device_policy.startswith("cuda") and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
    dtype = "float16"
dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
assert total_batch_size % micro_batch_size == 0, f"Total batch size: {total_batch_size} must be divisible by micro batch size: {micro_batch_size}"
for p in [data_file, prompt_template_file]:
    assert os.path.exists(p), f"File {p} does not exist"

# print config
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")

# -------------------------------------------------------------#
# Load tokenizer and prompt template
# -------------------------------------------------------------#
print("Initializing the tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading prompt template from {prompt_template_file}...")
with open(prompt_template_file, "r") as f:
    prompt_template = f.read()
prompt_template += tokenizer.eos_token if tokenizer.eos_token is not None else ""
pretty_print(prompt_template, title="Prompt template")

# -------------------------------------------------------------#
# Initialize wandb
# -------------------------------------------------------------#
print(f"Initializing wandb project {wandb_project} and run {wandb_run_name}...")
wandb.init(project=wandb_project, name=wandb_run_name, config=input_config)

# x/y-axis for train and eval steps
wandb.define_metric("step_train")
wandb.define_metric("step_eval")
wandb.define_metric("train/*", step_metric="step_train")
wandb.define_metric("eval/*", step_metric="step_eval")

# -------------------------------------------------------------#
# Initialize the models
# -------------------------------------------------------------#

# Policy model (trainable)
pretty_print("", title="Loading models")
print("Loading policy model...")
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=dtype, 
    attn_implementation=attention_type, 
    device_map=device_policy
)

# Reference model (frozen)
print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=dtype, 
    attn_implementation=attention_type, 
    device_map=device_ref
)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

if use_compile:
    policy_model = torch.compile(policy_model)
    ref_model = torch.compile(ref_model)

# -------------------------------------------------------------#
# Setup the datasets
# -------------------------------------------------------------#
pretty_print("", title="Setting up datasets")
train_dataset = DpoFinetuneDataset(
    dataset_path=data_file,
    shuffle=True,
    split="train",
    num_val=num_val,
    seed=seed
)

val_dataset = DpoFinetuneDataset(
    dataset_path=data_file,
    shuffle=True,
    split="val",
    num_val=num_val,
    seed=seed
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
assert len(train_dataset) > 0, "Train dataset is empty"
assert len(val_dataset) > 0, "Val dataset is empty"

# -------------------------------------------------------------#
# Setup the optimizer
# -------------------------------------------------------------#
if max_steps <= 0:
    max_steps = num_epochs * math.ceil(len(train_dataset) / total_batch_size)
warmup_steps = int(max_steps * warmup_ratio)
print(f"Warmup steps: {warmup_steps}")
print(f"Max. steps: {max_steps}")


pretty_print("", title="Initializing the optimizer")
optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=max_lr)
print(optimizer)


# -------------------------------------------------------------#
# Helper functions
# -------------------------------------------------------------#
def get_lr(step):
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr

@torch.no_grad()
def evaluate_val_data(policy_model, val_dataset, prompt_template, tokenizer):
    """Evaluate the model on the validation data"""
    policy_model.eval()
    start_time = time.time()
    
    total_correct = 0
    total_examples = 0
    
    for i in range(len(val_dataset)):
        # if i % 100 == 0:
        #     print(f"Evaluating example {i} of {len(val_dataset)}")
        
        example = val_dataset[i]
        correct = dpo_pref_accuracy(
            policy_model=policy_model,
            tokenizer=tokenizer,
            prompt=example['prompt'],
            chosen_response=example['chosen_response'],
            rejected_response=example['rejected_response'],
            prompt_template=prompt_template
        )
        total_correct += int(correct)
        total_examples += 1
    
    accuracy = total_correct / total_examples
    policy_model.train()
    return {"eval_accuracy": accuracy, "eval_time": time.time() - start_time}

def save_checkpoint(policy_model, tokenizer, step):
    ckpt_dir = output_dir + f"/checkpoint_{step}"
    os.makedirs(ckpt_dir, exist_ok=True)
    policy_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Saved checkpoint to {ckpt_dir}")

# -------------------------------------------------------------#
# Training loop
# -------------------------------------------------------------#
pretty_print("", title="Starting the training loop")

# Create iterator over training data
train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)
train_idx = 0
best_val_acc = -float('inf')

policy_model.train()

for step in range(max_steps):
    t0 = time.time()
    
    # Evaluate the model on the validation data
    if step % eval_interval == 0 or step == max_steps - 1:
        eval_dict = evaluate_val_data(policy_model, val_dataset, prompt_template, tokenizer)
        torch.cuda.empty_cache()
        print(f"eval_step: {step:04d} | eval_accuracy: {eval_dict['eval_accuracy']:.4f} | dt: {eval_dict['eval_time']:.2f}s")
        wandb.log({"step_eval": step, "eval/accuracy": eval_dict['eval_accuracy'], "eval/dt": eval_dict['eval_time']})
        if eval_dict['eval_accuracy'] > best_val_acc:
            best_val_acc = eval_dict['eval_accuracy']
            print(f"New best validation accuracy: {best_val_acc:.4f} at step {step}...")
            save_checkpoint(policy_model, tokenizer, step)
    
    # Zero gradients
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    
    for micro_step in range(grad_acc_steps):
        # Get next example
        if train_idx >= len(train_indices):
            random.shuffle(train_indices)
            train_idx = 0
        
        example = train_dataset[train_indices[train_idx]]
        train_idx += 1
        
        # Compute DPO loss
        with torch.autocast(device_type=device_type, dtype=dtype):
            loss = per_instance_dpo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                prompt=example['prompt'],
                chosen_response=example['chosen_response'],
                rejected_response=example['rejected_response'],
                prompt_template=prompt_template,
                beta=beta
            )
        
        # Scale the loss by gradient accumulation steps
        loss = loss / grad_acc_steps
        loss_accum += loss.detach()
        
        # Backward pass
        loss.backward()
    
    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_norm_clip)
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    # Optimizer step
    optimizer.step()
    
    # Log training metrics
    dt = time.time() - t0
    print(f"step: {step:04d} | train_loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}s")
    wandb.log({"step_train": step, "train/loss": loss_accum.item(), "train/lr": lr, "train/norm": norm.item(), "train/dt": dt})
    
    # Clear cache
    if step % cache_clear_interval == 0:
        torch.cuda.empty_cache()

print("Training complete!")
wandb.finish()

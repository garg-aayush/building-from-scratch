# simple run
# python train_gpt2.py
# ddp run
# torchrun --nproc_per_node=<num_gpus> train_gpt2.py
# NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<num_gpus> train_gpt2.py # in case you get ddp error, this helped me run script on RTX6000 ada comsumer gpus

import math
import os
import time

import numpy as np
import tiktoken
import torch
# for ddp training 
import torch.distributed as dist
import torch.nn.functional as F
# for logging
import wandb
from hellaswag import iterate_examples, render_example
from model import GPT, GPTConfig
from torch.nn.parallel import DistributedDataParallel as DDP

# -------------------------------------------------------------#
# params
# -------------------------------------------------------------#
wandb_project = "pre-training" # wandb project name
wandb_run_name = "gpt2-swiglu" # wandb run name
data_root = "/workspace/shards" # data root
ckpt_dir = "/workspace/ckpt" # checkpoint directory
eval_interval = 250      # (steps) interval for validation and hellaSwag evaluation
log_interval = 1         # (steps) interval for logging
grad_norm_clip = 1.0     # global norm gradient clipping
# data
total_batch = 524288    # 2^19, ~0.5M tokens, matching GPT-3 124M model
B = 64                 # batch size
T = 1024                # sequence length
# optimizer hyperparameters
max_lr = 1.5e-3           # maximum learning rate
min_lr = max_lr * 0.1   # minimum learning rate
warmup_steps = 300      # number of warmup steps, this is from original GPT-3 paper, and is too conservative, we can even go with like 100 steps
max_steps = 19073       # total number of steps, FineWeb-Edu 10B tokens (1 epoch training 10B/ 2^19)
weight_decay = 0.1      # weight decay for optimizer
betas = (0.9, 0.95)     # betas for optimizer
# model
vocab_size = 50304     # vocabulary size 50,000 merges + 256 byte pieces + 1 <endoftext> token -> nice number: 50,304
n_layer = 12           # number of layers
n_embd = 768           # embedding dimension
n_head = 12            # number of attention heads
# system
device = "cuda"        # device to use, "cuda" or "mps" or "cpu" (DDP only for "cuda")
seed = 42              # seed for the random number generator
data_seed = 1337       # seed for the data shuffle
use_compile = True    # use torch.compile to further speedup the model
# eval
val_loss_steps = 20        # number of steps for validation loss
num_return_sequences = 4    # number of return sequences
max_seq_len = 32            # maximum sequence length
start_seq = "Hello, I'm a language model," # start sequence
run_validation = True      # flag for running validation
run_hellaswag = True      # flag for running hellaswag
run_gen_samples = False      # flag for running generation samples
# H100 SXM: 989e12 flops (bf16)
# H100 PCIe: 756e12 flops (bf16)
# A100: 312e12 flops (bf16)
# RTX6000 Ada: 364e12 flops (bf16)
flops_promised = 989e12    # flops (bf16) promised by the gpu
# -------------------------------------------------------------#
# print config keys, cool way to see the config
config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool))}
for k,v in config.items():
    print(f"{k:<20}: {v}")
# -------------------------------------------------------------#

# -------------------------------------------------------------------------#
# data loader
# -------------------------------------------------------------------------#
seed_offset = 0
class TrainDataLoaderLite:
    
    def __init__(self, data_root, T, B, split="train", num_processes=1, process_rank=0, seed=42):
        
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.rng = np.random.default_rng(seed+seed_offset)

        # get the shards filenames
        self.shards = [s for s in os.listdir(data_root) if self.split in s]
        self.shards = sorted(self.shards)
        self.shards = [os.path.join(data_root, s) for s in self.shards]
        print(f"{split}: {len(self.shards)} shard(s)")

        # Memory-map shards so they don’t fully load into RAM
        self.mem = [np.load(f, mmap_mode='r') for f in self.shards]
        self.shard_lengths = [m.shape[0] for m in self.mem]
        
        # Build global window index ---
        self._build_index()
        self.ptr = 0

    def _build_index(self):
        """Build a list of all (shard_id, start_offset) windows."""
        all_indices = []
        for sid, L in enumerate(self.shard_lengths):
            # number of windows in this shard (discard leftover < T+1 tokens)
            max_windows = (L - (self.T + 1)) // self.T
            if max_windows <= 0:
                continue
            starts = (np.arange(max_windows) * self.T).astype(np.int64)
            shard_ids = np.full_like(starts, sid, dtype=np.int64)
            pairs = np.stack([shard_ids, starts], axis=1)
            all_indices.append(pairs)

        all_indices = np.concatenate(all_indices, axis=0)
        print(all_indices.shape)

        if self.split == "train":
            # shuffle and split across ranks (each GPU sees unique slice)
            self.rng.shuffle(all_indices)
            self.index = all_indices[self.process_rank::self.num_processes]
        else:
            # validation: no shuffle, same across ranks
            self.index = all_indices

        print(f"{self.split} index size: {len(self.index)} windows")

    def __len__(self):
        """Number of full batches in the dataset."""
        return len(self.index) // self.B

    def get_batch(self):
        """Return one batch of shape (B, T) for x and y."""
        rows = self.index[self.ptr : self.ptr + self.B]
        self.ptr += self.B

        xs, ys = [], []
        for sid, start in rows:
            # slice (T+1) tokens from shard
            t = self.mem[sid][start : start + self.T + 1].astype(np.uint16)
            t = torch.from_numpy(t)
            xs.append(t[:-1])
            ys.append(t[1:])
        x = torch.stack(xs).long()
        y = torch.stack(ys).long()
        return x, y

    def reset(self, seed=None):
        """Reset pointer and reshuffle (train only)."""
        if self.split == "train":
            if seed is not None:
                self.rng = np.random.default_rng(seed+seed_offset)
            self._build_index()
        self.ptr = 0

class ValDataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root='shards', seed=42):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.rng = np.random.RandomState(seed+seed_offset)
        self.eot = enc._special_tokens["<|endoftext|>"]  # end of text token
        
        # get the shards filenames
        shards = [s for s in os.listdir(data_root) if self.split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        # shuffle the shards
        self.rng.shuffle(self.shards)
        assert len(shards) > 0, f"No shards found for split: {self.split}"
        
        # load the dataset
        self.cur_shard = 0
        self.tokens = self._load_tokens(self.shards[self.cur_shard])
        self.cur_pos = self.process_rank * (self.B * self.T)

        
    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # input to the model
        y = buf[1:].view(B, T)  # output of the model
        # advance position
        self.cur_pos += B * T * self.num_processes
        
        # if loading next batch is out of bounds, load the next shard
        if self.cur_pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.cur_shard])
            self.cur_pos = self.process_rank * (self.B * self.T)
        
        return x, y
    
    def _load_tokens(self, filename):
        # memory mapping for efficiency
        np_tensor = np.load(filename, mmap_mode='r')
        # For validation split, return tokens as-is without shuffling
        if self.split == "val":
            return torch.tensor(np_tensor, dtype=torch.long)
        else:
            # For training split, shuffle documents to reduce temporal patterns
            t = torch.tensor(np_tensor, dtype=torch.long)
            # Split the token sequence into individual documents at end-of-text markers
            doc_breaks = (t == self.eot).nonzero().flatten().tolist()
            docs, start = [], 0 
            # Extract each document including its EOT token
            for end in doc_breaks:
                docs.append(t[start:end+1])  # include EOT
                start = end + 1
            if start < len(t):  # last doc without trailing EOT
                docs.append(t[start:])
            # Randomly shuffle the order of documents to break temporal patterns
            self.rng.shuffle(docs)
            # Concatenate all shuffled documents back into a single tensor
            return torch.cat(docs, dim=0)
    def reset(self):
        self.cur_shard = 0
        self.tokens = self._load_tokens(self.shards[self.cur_shard])
        self.cur_pos = self.process_rank * (self.B * self.T)
# -------------------------------------------------------------------------#


# -------------------------------------------------------------------------#
# helper functions
# -------------------------------------------------------------------------
# copied from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
# HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# cosine decay learning-rate scheduler with warmup
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

# -------------------------------------------------------------#
assert device in ["cuda", "mps", "cpu"], f"Invalid device: {device}"
if device == "cuda" and torch.cuda.is_available():
    device = "cuda"
    device_type = 'cuda'
elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    device_type = 'mps'
else:
    device = "cpu"
    device_type = 'cpu'
print(f"Using device: {device}")


# setup DDP training
# torchrun commands sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
# and we can use them to initialize the DDP
ddp = int(os.environ.get("RANK", -1)) != -1 # if RANK is not -1, then we are using DDP
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"]) # global rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # local rank on a single node
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # total number of processes
    device = f"cuda:{ddp_local_rank}"
    device_type = 'cuda'
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0 # this process will do the printing, logging, checkpointing, etc.
    seed_offset = ddp_rank
    # sanity check
    print(f"Using DDP with rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size} on device {device}")
else:
    # vanilla single GPU/CPU/MPS training
    master_process = True
    ddp_rank = 0
    seed_offset = 0
    ddp_local_rank = 0
    ddp_world_size = 1

# set the seed, different for each process
torch.cuda.manual_seed(seed + seed_offset)
if device_type == "cuda":  # Fixed: only set CUDA seed if using CUDA
    torch.cuda.manual_seed(seed + seed_offset)

# use tf32
torch.set_float32_matmul_precision("high")

# create wandb run and ckpt dir
if master_process:
    os.makedirs(ckpt_dir, exist_ok=True)
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    print(f"Logging to wandb, project: {wandb_project}, run: {wandb_run_name}")

# get a data batch
print("Calculate gradient accumulation steps...")
assert total_batch % (B * T * ddp_world_size) == 0, f"Total batch size {total_batch} is not divisible by B*T*WORLD_SIZE={B * T * ddp_world_size}"
grad_accum_steps = total_batch // (B * T * ddp_world_size)
print(f"Total desired batch size: {total_batch}")
print(f"gradient accumulation steps: {grad_accum_steps}")

# create the encoder
print("Creating encoder...")
enc = tiktoken.get_encoding("gpt2")

# create the data loaders
train_loader = TrainDataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root=data_root, seed=data_seed)
val_loader = ValDataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root=data_root, seed=data_seed)


# initialize the model
print("Initializing model...")
model_args = dict(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head, block_size=T)
model = GPT(GPTConfig(**model_args))
model.to(device)

# setup the optimizer
print("Setting up optimizer...")
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, betas=betas, device=device_type)

# use torch.compile to further speedup the model
if use_compile:
    print("Compiling model...")
    model = torch.compile(model)

print("Moving model to device...")
if ddp:
    # pass ddp_local_rank to the model to ensure the model is moved to the correct device
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

@torch.no_grad()
def estimate_loss():
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    for _ in range(val_loss_steps):
        x, y = val_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        val_loss_accum += loss.detach()
    val_loss_accum /= val_loss_steps
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    model.train()
    return val_loss_accum.item()

@torch.no_grad()
def estimate_hella_acc():
    model.eval()
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    model.train()
    return acc_norm, num_correct_norm, num_total

@torch.no_grad()
def generate_samples():
    model.eval()
    
    # prefix the tokens
    tokens = enc.encode(start_seq)  # 8 tokens
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8) - Fixed comment
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed + seed_offset)
    
    # generate the text -> x: (B,T) where B=5, T=8
    while xgen.size(1) < max_seq_len:
        # forward the model to get the logits
        logits, _ = model(xgen)  # (B,T,vocab_size)
        # logits at last position (inefficient but correct)
        logits = logits[:, -1, :]  # (B, vocab_size)
        # calculate probabilities
        probs = F.softmax(logits, dim=-1)
        # do topk sampling of 50 (default in HF pipeline)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probs
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B,1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        xgen = torch.cat([xgen, xcol], dim=1)
    
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_seq_len].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank}, sample {i}: {decoded}")
    model.train()

best_val_loss = 1e9
running_mfu = -1.0
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    log_dict = {"step": step}
    
    # validation
    if (step % eval_interval == 0 or last_step) and run_validation:
        val_loss = estimate_loss()
        if master_process:
            print(f"validation loss: {val_loss:.4f}")
            log_dict["val/loss"] = val_loss

            # save the best model
            if (val_loss < best_val_loss or last_step) and step > 0:
                best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "model_args": model_args,
                    "config": config
                }
                print(f"Saving model at step {step}, val_loss: {val_loss:.4f} -> best_val_loss: {best_val_loss:.4f}")
                ckpt_name = "final_model.pt" if last_step else "best_model.pt"
                torch.save(checkpoint, os.path.join(ckpt_dir, ckpt_name))
    
    # hellaswag evaluation
    if (step % eval_interval == 0 or last_step) and run_hellaswag:
        hella_acc_norm, num_correct_norm, num_total = estimate_hella_acc()
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={hella_acc_norm:.4f}")
            log_dict["val/hella_norm"] = hella_acc_norm
        
    # generate samples from the model (except at step 0)
    if (step % eval_interval == 0 or last_step) and run_gen_samples:
        generate_samples()
    
        
    # training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # accumulate gradients over multiple steps
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            # only synchronize on the final micro-step and all-reduce the loss_accum across all processes
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # use bfloat16 for the model forward pass, supported on Ampere and above
        # note since we are using bf16 and not f16, we don't need to use gradient scaler
        # As bf16 has the same range as fp32
        # Karpathy suggests to only refer to https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale down the loss by the number of gradient accumulation steps 
        # because the gradients just add up on each successive step (loss.backward())
        # and we want mean instead of sum
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        # all-reduce the loss_accum across all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    # global norm gradient clipping at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
    
    # determine and set the learning rate for the current step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    # torch.cuda.synchronize() to ensure the GPU finishes before timing
    torch.cuda.synchronize()
    
    # log the training loss
    t1 = time.time()
    dt = (t1 - t0) * 1000 # convert to ms
    # tokens per second is a better metric than dt because it is independent of the batch size and sequence length
    tokens_per_second = (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size / (t1-t0)
    if master_process and (step % log_interval == 0 or last_step):
        if step > 5: # let the training loop stabilize
            mfu = raw_model.estimate_mfu(grad_accum_steps * B, dt/1000, flops_promised)
            # smooth the mfu (moving average)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"step: {step:04d}, loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f} | mfu: {running_mfu*100:.2f}%")
        train_log_dict = {
            "train/loss": loss_accum.item(), 
            "train/lr": lr, 
            "train/norm": norm.item(), 
            "dt": dt,
            "mfu": running_mfu,
            "tok/s": tokens_per_second
        }
        # add the train/log_dict to the log_dict
        log_dict.update(train_log_dict)
        wandb.log(log_dict)
            
if ddp:
    dist.destroy_process_group()
# simple run
# python train_gpt2.py
# ddp run
# NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<num_gpus> train_gpt2.py

import math
from dataclasses import dataclass
import os
import tiktoken
import torch
import numpy as np
from hellaswag import render_example, iterate_examples

# Set NCCL environment variable for DDP stability
os.environ['NCCL_P2P_DISABLE'] = '1'
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
import time
import inspect

# for ddp training 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------#
# GPTConfig: Configuration for the GPT-2 model
# -----------------------------------#
@dataclass
class GPTConfig:
    block_size: int = 1024  # max seq. length
    vocab_size: int = (
        50257  # num. of tokens: 50,000 merges + 256 byte pieces + 1 <endoftext> token
    )
    n_layer: int = 12  # number of layers
    n_embd: int = 768  # embedding dimension
    n_head: int = 12  # number of attention heads


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), f"n_embd must be divisible by n_head: {config.n_embd} % {config.n_head} != 0"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, n_embd
        qkv = self.c_attn(x)
        # split into q, k, v, size: (B, T, 3 * n_embd) -> (B, T, n_embd) * 3
        q, k, v = qkv.split(self.n_embd, dim=2)
        # rearrange to (B, nh, T, hs), mimics multi-head attention in the original paper
        k = rearrange(k, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)
        q = rearrange(q, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)
        v = rearrange(v, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)

        # # attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = einsum(q, k, "B nh T1 hs, B nh T2 hs -> B nh T1 T2") * (
        #     1.0 / math.sqrt(k.size(-1))
        # )
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = att @ v
        # use FlashAttention 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # re-assemble all head outputs side by side
        y = rearrange(y, "B nh T hs -> B T (nh hs)")
        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # transformer block: reduce-map operation
        # attention: reduce/communication operation
        x = x + self.attn(self.ln_1(x))
        # mlp: map/thinking operation, here individual tokens think about the information they gathered and do not communicate with each other
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # final classification head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        # it points to the same memory address, now we are training approximately 30% less parameters
        self.transformer.wte.weight = self.lm_head.weight

        # initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 0.02 is roughly in range of Xavier initialization. As Xavier initialization is 1/sqrt(n_in), so for n_in = [768-1600], the std is ~ 0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # according to GPT-2 paper, we need to scale down the weights by 1/sqrt(2*n_layer) to control the growth of activations inside the residual stream in the forward pass
                std = std * (1 / math.sqrt(2 * self.config.n_layer))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: token indices
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the tokens and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, n_embd)
        x = pos_emb + tok_emb
        # forward pass through the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward pass through the final layer norm and classifier
        x = self.transformer.ln_f(x)
        # every B,T calculate the logits for what token comes next in the sequence
        logits = self.lm_head(x)  # (B,T,vocab_size)
        loss = None
        if targets is not None:
            # cross-entropy function does not like multi-dimensional inputs, so we need to flatten the logits and targets
            # logits: (B,T,vocab_size) -> (B*T,vocab_size)
            # targets: (B,T) -> (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained weights from Hugging Face."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all the parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # separate into decay and non-decay parameters
        # Any parameter that has a dimension greater than or equal to 2 is a weight/matrix parameter (matmuls, embeddings, etc.) that should be decayed, while all biases and other 1D (layerNorm gains, etc.) parameters should not be decayed
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        # create optimizers
        optim_groups = [
            {"params": [p for p in decay_params], "weight_decay": weight_decay},
            {"params": [p for p in nodecay_params], "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decay tensors: {len(decay_params)} and parameters: {num_decay_params:,}")
        print(f"Number of non-decay tensors: {len(nodecay_params)} and parameters: {num_nodecay_params:,}")
        # create AdamW optimizer and enable fused AdamW implementation when available
        # fused AdamW implementation is available on later versions of PyTorch and saves overhead as instead of updating each parameter individually, it updates them in a single kernel
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -------------------------------------------------------------------------#
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root='~/data/shards'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"], f"Invalid split: {split}"
        
        # get the shards filenames
        shards = [s for s in os.listdir(data_root) if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split: {split}"

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
        np_tensor = np.load(filename)
        return torch.tensor(np_tensor, dtype=torch.long)
    
    def reset(self):
        self.cur_shard = 0
        self.tokens = self._load_tokens(self.shards[self.cur_shard])
        self.cur_pos = self.process_rank * (self.B * self.T)
# -------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# copied from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
# helper function for HellaSwag eval
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

# setup DDP training
# torchrun commands sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
# and we can use them to initialize the DDP
ddp = int(os.environ.get("RANK", -1)) != -1 # if RANK is not -1, then we are using DDP
if ddp:
    assert torch.cuda.is_available(), "CUDA is not available, we cannot use DDP"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"]) # global rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # local rank on a single node
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # total number of processes
    device = f"cuda:{ddp_local_rank}"
    print(f"Using DDP with rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size} on device {device}")
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0 # this process will do the printing, logging, checkpointing, etc.
else:
    # vanilla single GPU/CPU/MPS training
    master_process = True
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    # get available device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        device_type = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_type = 'mps'
    print(f"Using device: {device}")

# create new variable device_type
device_type = 'cuda' if 'cuda' in device else 'cpu'
torch.manual_seed(42)
if device_type == "cuda":  # Fixed: only set CUDA seed if using CUDA
    torch.cuda.manual_seed(42)

# get a data batch
total_batch = 524288 # 2^19, ~0.5M tokens
B = 32
T = 1024
assert total_batch % (B * T * ddp_world_size) == 0, f"Total batch size {total_batch} is not divisible by B*T*WORLD_SIZE={B * T * ddp_world_size}"
grad_accum_steps = total_batch // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

print(f"DDP rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root="/root/data/shards")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_root="/root/data/shards")

# use tf32
torch.set_float32_matmul_precision("high")

# create the model
# use a "nice" number for the vocabulary size
model = GPT(GPTConfig(vocab_size=50304))  # random model initialization
print("Model loaded successfully")
model.to(device)

# optimize
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # this is from original GPT-3 paper, and is too conservative, we can even go with like 100 steps
max_steps = 19073  # 10BT/2^19
# cosine decay learning-rate schedule with warmup
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
# update the optimizer to use the same hyperparameters as GPT-3
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device_type)

# use torch.compile to further speedup the model
print("Compiling model...")
model = torch.compile(model)
print("Model compiled successfully")

print("Moving model to device...")
if ddp:
    # pass ddp_local_rank to the model to ensure the model is moved to the correct device
    model = DDP(model, device_ids=[ddp_local_rank])
print("Model moved to device successfully")

# create the encoder
enc = tiktoken.get_encoding("gpt2")

# create log directory
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # validation
    if step % 10 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.get_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss.detach()
            val_loss_accum /= val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val_loss: {val_loss_accum.item():.4f}\n")
    
    # hellaswag
    if (step % 10 == 0 or last_step):
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
            with torch.no_grad():
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
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella_norm: {acc_norm:.4f}\n")
                

    # generate samples from the model (except at step 0)
    if step % 10 == 0 or last_step:
        model.eval()
        num_return_sequences = 4
        max_seq_len = 32
        start_seq = "Hello, I'm a language model,"
        # prefix the tokens
        tokens = enc.encode(start_seq)  # 8 tokens
        tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8) - Fixed comment
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        
        # generate the text -> x: (B,T) where B=5, T=8
        while xgen.size(1) < max_seq_len:
            # forward the model to get the logits
            with torch.no_grad():
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
    
    # training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # accumulate gradients over multiple steps
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
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
        if ddp:
            # only synchronize on the final micro-step and all-reduce the loss_accum across all processes
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        # all-reduce the loss_accum across all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # global norm gradient clipping at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for the current step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    # torch.cuda.synchronize() to ensure the GPU finishes before timing
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # convert to ms
    # tokens per second is a better metric than dt because it is independent of the batch size and sequence length
    tokens_per_second = (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size / (t1-t0)
    if master_process:
        print(f"step: {step:04d}, loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train_loss: {loss_accum.item():.4f}\n")
            
if ddp:
    dist.destroy_process_group()
import math
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
import time
import inspect

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
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load the dataset
        # data has approximately 40K lines, 200K words, 1M bytes
        with open("data/input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch has {len(self.tokens) // (self.B * self.T)} batches")

        # set state
        self.cur_pos = 0

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # input to the model
        y = buf[1:].view(B, T)  # output of the model
        # advance position
        self.cur_pos += B * T
        # if loading past the end of the dataset, loop back to the beginning
        if self.cur_pos + B * T + 1 > len(self.tokens):
            self.cur_pos = 0
        return x, y


# -------------------------------------------------------------------------#
num_return_sequences = 5
max_seq_len = 30
start_seq = "Hello, I'm a language model,"

# get available device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# get a data batch
total_batch = 524288 # 2^19, ~0.5M tokens
B = 16
T = 1024
assert total_batch % (B * T) == 0, f"Total batch size {total_batch} is not divisible by B*T={B * T}"
grad_accum_steps = total_batch // (B * T)
print(f"Total desired batch size: {total_batch}")
print(f"gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

# use tf32
torch.set_float32_matmul_precision("high")
# get logits
# use a "nice" number for the vocabulary size
model = GPT(GPTConfig(vocab_size=50304))  # random model initialization
print("Model loaded successfully")
model.to(device)
# use torch.compile to further speedup the model
model = torch.compile(model)

# optimize
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
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
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    # accumulate gradients over multiple steps
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        # use bfloat16 for the model forward pass, supported on Ampere and above
        # note since we are using bf16 and not f16, we don't need to use gradient scaler
        # As bf16 has the same range as fp32
        # Karpathy suggests to only refer to https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale down the loss by the number of gradient accumulation steps 
        # because the gradients just add up on each successive step (loss.backward())
        # and we want mean instead of sum
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
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
    tokens_per_second = (train_loader.B * train_loader.T) * grad_accum_steps / (t1-t0) 
    print(f"step: {step:04d}, loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/s: {tokens_per_second:.2f}")

import sys

sys.exit(0)
# set to eval mode and move to appropriate device
model.eval()
model.to(device)

# prefix the tokens
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(start_seq)  # 8 tokens
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8) - Fixed comment
x = tokens.to(device)

# generate the text -> x: (B,T) where B=5, T=8
torch.manual_seed(42)
if device == "cuda":  # Fixed: only set CUDA seed if using CUDA
    torch.cuda.manual_seed(42)
while x.size(1) < max_seq_len:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)  # (B,T,vocab_size)
        # logits at last position (inefficient but correct)
        logits = logits[:, -1, :]  # (B, vocab_size)
        # calculate probabilities
        probs = F.softmax(logits, dim=-1)
        # do topk sampling of 50 (default in HF pipeline)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probs
        ix = torch.multinomial(topk_probs, 1)  # (B,1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat([x, xcol], dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_seq_len].tolist()
    decoded = enc.decode(tokens)
    print(decoded)
    print("-" * 100)

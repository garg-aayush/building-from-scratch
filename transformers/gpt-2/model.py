import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einsum import einsum, rearrange


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

        # attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = einsum(q, k, "B nh T1 hs, B nh hs T2 -> B nh T1 T2") * (
            1.0 / math.sqrt(k.size(-1))
        )
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
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

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # transformer block: reduce-map operation
        # attention: reduce/communication operation
        x = x + self.attn(self.ln1(x))
        # mlp: map/thinking operation, here individual tokens think about the information they gathered and do not communicate with each other
        x = x + self.mlp(self.ln2(x))
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

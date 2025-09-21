import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from transformers import GPT2LMHeadModel


# -----------------------------------#
# GPTConfig: Configuration for the GPT-2 model
# -----------------------------------#
@dataclass
class GPTConfig:
    block_size: int = 1024  # max seq. length
    vocab_size: int = 50257  # num. of tokens: 50,000 merges + 256 byte pieces + 1 <endoftext> token
    n_layer: int = 12  # number of layers
    n_embd: int = 768  # embedding dimension
    n_head: int = 12  # number of attention heads

# RoPE module that rotates query/key feature pairs by position-dependent angles
class RotaryEmbedding(nn.Module):  
    def __init__(self, dim, max_seq_len=2048):  
        super().__init__()
        # dim: per-head size; max_seq_len: how many positions to precompute
        assert dim % 2 == 0, f"RotaryEmbedding requires even head_dim, got {dim}"
        self.dim = dim
        self.max_seq_len = max_seq_len
        # inverse frequencies for each pair
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # keep on device, not trainable
        # positions [0..max_seq_len-1]
        t = torch.arange(self.max_seq_len).type_as(self.inv_freq)
        # outer product → angles per (position, pair)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # duplicate to match two halves of the head dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        # cache sin/cos for all positions/pairs, shaped [1,1,T,D]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x):
        # x: [bs, num_heads, seq_len, head_dim];
        seq_len = x.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)

        def rotate_half(x_):
            # rotate by swapping halves and negating the second
            x1 = x_[..., : self.dim // 2]
            x2 = x_[..., self.dim // 2 :]
            # [-x2, x1]: a 90-degree rotation in each 2D feature pair
            return torch.cat((-x2, x1), dim=-1)
        
        # standard RoPE: apply rotation with per-position cos/sin
        return x * cos + rotate_half(x) * sin

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
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.rope = RotaryEmbedding(dim=self.n_embd // self.n_head, max_seq_len=config.block_size)


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, n_embd
        qkv = self.c_attn(x)
        # split into q, k, v, size: (B, T, 3 * n_embd) -> (B, T, n_embd) * 3
        q, k, v = qkv.split(self.n_embd, dim=2)
        # rearrange to (B, nh, T, hs), mimics multi-head attention in the original paper
        k = rearrange(k, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)
        q = rearrange(q, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)
        v = rearrange(v, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)

        # apply rope
        q = self.rope(q)
        k = self.rope(k)
        
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


# SwiGLU: https://arxiv.org/pdf/2002.05202
class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig, factor: float = 8/3):
        super().__init__()
        # Two linear projections (for swiglu)
        self.c_fc1 = nn.Linear(config.n_embd, int(config.n_embd * factor))
        self.c_fc2 = nn.Linear(config.n_embd, int(config.n_embd * factor))
        # Output projection back to input_dim
        self.c_proj = nn.Linear(int(config.n_embd * factor), config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # SwiGLU: ((xW1) * swish(xW2)) * W3
        x_proj = self.c_fc1(x)
        gate = F.silu(self.c_fc2(x))
        x = self.c_proj(x_proj * gate)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = SwiGLU(config)

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
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the tokens and position embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, n_embd)
        x = tok_emb
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
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer
    
    def get_num_parameters(self, non_embeddings=True):
        num_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            # as wte are shared with lm_head, we can subtract them from the total
            num_params -= self.transformer.wte.weight.numel()
        return num_params

    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised=989e12):
        """Estimate flops utilization (MFU) in ratio"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        
        # number of params
        N = self.get_num_parameters()
        # flops per iteration
        L, H, Q, T = self.config.n_layer, self.config.n_head, self.config.n_embd//self.config.n_head, self.config.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwd_bwd = flops_per_token * T
        flops_per_iter = flops_per_fwd_bwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = flops_promised
        mfu = flops_achieved / flops_promised
        return mfu
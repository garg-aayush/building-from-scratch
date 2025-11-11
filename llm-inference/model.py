# Baseline Model implementation taken from commit: a100995

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
    eos_token_id: int = 50256 # <|endoftext|> token id


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
        self.block_size = config.block_size
        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        # for KV-cache
        # persistent=False means that the buffer is not saved to the state_dict
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x, use_cache=False, attn_mask_padding=None):
        B, T, C = x.size()  # batch size, sequence length, n_embd
        qkv = self.c_attn(x)
        # split into q, k, v, size: (B, T, 3 * n_embd) -> (B, T, n_embd) * 3
        q, k, v = qkv.split(self.n_embd, dim=2)
        # rearrange to (B, nh, T, hs), mimics multi-head attention in the original paper
        k = rearrange(k, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)
        q = rearrange(q, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)
        v = rearrange(v, "B T (nh hs) -> B nh T hs", nh=self.n_head)  # (B, nh, T, hs)

        # Apply kv cache
        # pre-allocate memory for the cache
        if use_cache:
            if self.k_cache is None:
                self.k_cache = torch.zeros(B, self.n_head, self.block_size, self.n_embd // self.n_head, device=q.device)
                self.v_cache = torch.zeros_like(self.k_cache)
                self.cur_pos = 0
            
            # Handle sliding window when cache is full
            if self.cur_pos + T > self.block_size:
                overflow = self.cur_pos + T - self.block_size
                # Shift cache to the left by 'overflow' positions
                # This discards the oldest 'overflow' tokens
                self.k_cache[:, :, :-overflow] = self.k_cache[:, :, overflow:].clone()
                self.v_cache[:, :, :-overflow] = self.v_cache[:, :, overflow:].clone()
                # Adjust position to account for the shift
                self.cur_pos = self.block_size - T
            
            # Write new k,v at current position
            self.k_cache[:, :, self.cur_pos:self.cur_pos + T] = k
            self.v_cache[:, :, self.cur_pos:self.cur_pos + T] = v
            self.cur_pos += T
            
            # Return the valid cached portion (always full block_size when in sliding window mode)
            k, v = self.k_cache[:, :, :self.cur_pos], self.v_cache[:, :, :self.cur_pos]
        else:
            self.k_cache, self.v_cache = None, None
            self.cur_pos = 0
        Tq = q.size(2) # number of query tokens in the forward pass
        Tk = k.size(2) # number of key/value tokens in total (cached + forward pass)
        
        # # attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = einsum(q, k, "B nh T1 hs, B nh T2 hs -> B nh T1 T2") * (
        #     1.0 / math.sqrt(k.size(-1))
        # )
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = att @ v
        
        # If use_cache=False or num_query_tokens == num_key_tokens, use the simple version of FlashAttention
        if not use_cache or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # query has to attend to all the keys/values in the cache, is_casual=False as Tq == 1
        elif Tq == 1:
            # If there is only one query token, we can use the cached version of FlashAttention
            # Note, for current inference, this will be used when use_cache=True and is equivalent to else case for Tq == 1
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # when there are multiple query tokens and use_cache=True
        else:
            # True = keep, False = mask out
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            # apply casual attention
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            
            if attn_mask_padding is not None and not attn_mask_padding.all():
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Tq, Tk)
                attn_mask_padding = attn_mask_padding.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Tk)
                attn_mask = attn_mask & attn_mask_padding
            
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            
        # re-assemble all head outputs side by side
        y = rearrange(y, "B nh T hs -> B T (nh hs)")
        # output projection
        y = self.c_proj(y)

        return y
    
    def clear_cache(self):
        self.k_cache, self.v_cache = None, None
        self.cur_pos = 0


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

    def forward(self, x, use_cache=False, attn_mask_padding=None):
        # transformer block: reduce-map operation
        # attention: reduce/communication operation
        x = x + self.attn(self.ln_1(x), use_cache=use_cache, attn_mask_padding=attn_mask_padding)
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
        
        # for kv cache
        self.current_pos = 0

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

    def forward(self, idx, targets=None, use_cache=False, attn_mask_padding=None):
        # idx: token indices
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the tokens and position embeddings
        if use_cache:
            # Handle sliding window: when current_pos + T exceeds block_size, adjust position
            if self.current_pos + T > self.config.block_size:
                overflow = self.current_pos + T - self.config.block_size
                self.current_pos = self.config.block_size - T

            pos = torch.arange(self.current_pos, self.current_pos + T, dtype=torch.long, device=idx.device)  # shape (T)
            self.current_pos += T
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings (B, T, n_embd)
        x = pos_emb + tok_emb
        # forward pass through the transformer
        for block in self.transformer.h:
            x = block(x, use_cache=use_cache, attn_mask_padding=attn_mask_padding)
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

    def clear_kv_cache(self):
        self.current_pos = 0
        for block in self.transformer.h:
            block.attn.clear_cache()

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained weights from Hugging Face."""
        assert model_type in {"distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "distilgpt2": dict(n_layer=6, n_head=12, n_embd=768),  # 82M params
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
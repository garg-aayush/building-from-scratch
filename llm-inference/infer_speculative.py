# Speculative decoding implementation following the deepmind paper: https://arxiv.org/abs/2302.01318
# This implementation is adapted from: https://github.com/feifeibear/LLMSpeculativeSampling
import math
import time
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F
from model import GPT

# -------------------------------------------------------------------------#
# Input parameters
max_new_tokens = 200 # maximum number of new tokens to generate
temperature = 1.0 # temperature for sampling
top_k = 50 # top-k sampling (num. of highest prob vocab tokens to keep)
top_p = 0.9 # top-p sampling (cumulative probability threshold)
start_seq = "The following is a short story about a cat:" # start sequence
device = "cuda" # device to use
dtype = "float16" # "float16" or "bfloat16" or "float32"
target_model_name = "gpt2-large" # target model name for speculative decoding
draft_model_name = "gpt2" # draft model name for speculative decoding
use_speculative = True # use speculative decoding (True) or only use the target model (False)
gamma = 4 # number of draft model predictions to use
seed = 42 # seed for the random number generator
# -------------------------------------------------------------------------#

# ---------------- Initialize the model ---------------- #
if use_speculative:
    assert draft_model_name is not None, "Draft model name is required for speculative decoding and ideally should be a smaller model with same tokenization as the target model)"
    assert gamma in [3, 4, 5, 6, 7], "gamma must be a positive integer from 3 to 7"
    
# available device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# check if the device supports the dtype
if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
    dtype = "float16"
    print("Warning: bfloat16 is not supported on this device, using float16 instead")

# set the inference precision
pdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
ctx = nullcontext() if device == "cpu" or device == "mps" else torch.amp.autocast(device_type=device, dtype=pdtype)
print(f"Using device: {device} and dtype: {dtype} with context: {ctx}")

# set the seed
torch.manual_seed(seed)
if device == "cuda": 
    torch.cuda.manual_seed(seed)

# load the target and draft models
target_model = GPT.from_pretrained(target_model_name).eval().to(device)
print(f"Target model {target_model_name} loaded successfully")
if use_speculative:
    draft_model = GPT.from_pretrained(draft_model_name).eval().to(device)
    print(f"Draft model {draft_model_name} loaded successfully")

# ---------------- Initialize the tokenizer ---------------- #
enc = tiktoken.get_encoding("gpt2")

# ---------------- Encode the start sequence ---------------- #
tokens = enc.encode(start_seq, allowed_special={"<|endoftext|>"})  # n tokens (list of integers)
x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]  # (1, n)

# ---------------- helper functions ---------------- #
def run_sanity_checks(temperature, top_k, top_p):
    # handle temperature close to 0
    assert temperature > 0.0, "Temperature must be greater than 0"
    if top_k is not None:
        assert isinstance(top_k, int) and top_k > 0, "top_k must be a positive integer"
    if top_p is not None:
        assert isinstance(top_p, float) and top_p > 0 and top_p <= 1, "top_p must be a float between 0 and 1"
    if top_p is not None:
        assert isinstance(top_p, float), "top_p must be a float"

def norm_logits(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature < 0.1:
        logits = logits - logits.max(dim=-1, keepdim=True).values + 1
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        topk_probs, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits[logits < topk_probs[:, [-1]]] = -float('inf')
    if top_p is not None and top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[...,[0]] = False
        sorted_logits[sorted_indices_to_remove] = -float('inf')
        logits = torch.scatter(logits, -1, sorted_indices, sorted_logits)
    return logits

def sample(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def max_fn(x):
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum

# ---------------- Generate the text using only the target model (baseline)---------------- #
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0,
             top_k=None, top_p=None):
    
    # run sanity checks
    run_sanity_checks(temperature, top_k, top_p)

    # generate the text
    for i in range(max_new_tokens):
        # Pass full context (cropped if needed)
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits
        logits, _ = model(idx_cond)  # (B,T,vocab_size) idx_cond: (B,T)
        # logits at last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # normalize the logits (temperature scaling following by top-k and top-p sampling)
        logits = norm_logits(logits, temperature, top_k, top_p)
        # sample from the distribution
        idx_next = sample(logits)  # (B, 1)
        # append to the sequence
        idx = torch.cat([idx, idx_next], dim=1)
        # early stopping if the token is the EOS token
        if idx_next == model.config.eos_token_id:
            print("EOS token encountered, stopping generation")
            break
    return idx


@torch.no_grad()
def generate_speculative(target_model, draft_model, idx, max_new_tokens, gamma=5, temperature=1.0,
             top_k=None, top_p=None):
    
    # run sanity checks
    run_sanity_checks(temperature, top_k, top_p)
    
    # track tokens drafted and accepted
    total_drafted = 0
    total_accepted = 0
    
    initial_len = idx.shape[1]
    max_len = initial_len + max_new_tokens
    
    # Generate tokens in batches using speculative decoding
    # We use a for loop but need to account for generating multiple tokens per iteration
    for _ in range(max_new_tokens):
        # Check if we've generated enough tokens
        if idx.shape[1] >= max_len:
            break
            
        x = idx
        idx_len = idx.shape[1]
        
        # generate draft model predictions and logits one by one for gamma times
        draft_logits_list = []
        for _ in range(gamma):
            # handle the case where the idx is longer than the draft model's block size
            x_cond = x if x.size(1) <= draft_model.config.block_size else x[:, -draft_model.config.block_size:]
            # generate the logits
            draft_output, _ = draft_model(x_cond)
            # normalize the logits (temperature scaling following by top-k and top-p sampling)
            draft_logits_norm = norm_logits(draft_output[:, -1, :], temperature, top_k, top_p)
            # store the logits at the last position
            draft_logits_list.append(draft_logits_norm) 
            # sample the next token
            idx_next = sample(draft_logits_norm)
            x = torch.cat((x, idx_next), dim=1)
        
        # stack all draft logits: (gamma, batch, vocab) -> (batch, gamma, vocab)
        draft_logits = torch.stack(draft_logits_list, dim=1)
        
        # get target model predictions for the entire sequence (all at once)
        x_cond = x if x.size(1) <= target_model.config.block_size else x[:, -target_model.config.block_size:]
        target_output, _ = target_model(x_cond)
        # we need the logits for the last gamma tokens
        target_logits = target_output[:, -(gamma+1):, :]
        
        # normalize target logits
        for i in range(target_logits.shape[1]):
            target_logits[:, i, :] = norm_logits(target_logits[:, i, :], temperature, top_k, top_p)
        
        # rejection sampling loop
        is_all_accept = True
        n = idx_len - 1
        accepted_cur = 0
        for i in range(gamma):
            # sample a random number from the uniform distribution
            rand_num = torch.rand(1, device=target_output.device)
            # get the token at position i
            idx_i = x[:, idx_len + i]
            # compare probabilities at position i
            target_prob = F.softmax(target_logits[:, i, :], dim=-1)
            draft_prob = F.softmax(draft_logits[:, i, :], dim=-1)
            # accept or reject the token
            if rand_num < torch.min(torch.tensor([1.0], device=draft_output.device), target_prob[:, idx_i] / draft_prob[:, idx_i]):
                # accept the token
                n += 1
                accepted_cur += 1
            else:
                # reject: sample from the adjusted distribution (as per deepmind paper)
                adjusted_probs = max_fn(target_prob - draft_prob)
                t = torch.multinomial(adjusted_probs, num_samples=1)
                is_all_accept = False
                break
        
        total_drafted += gamma
        total_accepted += accepted_cur
        
        idx = x[:, :n + 1]
        
        if is_all_accept:
            # all tokens accepted, sample one more from the target model
            t = sample(target_logits[:, -1, :])
        
        idx = torch.cat((idx, t), dim=1)
        
        # Early stopping if EOS token is generated
        if idx[0, -1] == target_model.config.eos_token_id:
            print("EOS token encountered, stopping generation")
            break
    
    # Trim to max_new_tokens if speculative decoding generated more tokens than max_len
    if idx.shape[1] > max_len:
        idx = idx[:, :max_len]
    
    acceptance_ratio = total_accepted / total_drafted if total_drafted > 0 else 0
    return idx, acceptance_ratio

# print the generated text
with ctx:
    start_time = time.time()
    if use_speculative:
        y, acceptance_ratio = generate_speculative(target_model=target_model, draft_model=draft_model, idx=x, max_new_tokens=max_new_tokens, 
                gamma=gamma, temperature=temperature, top_k=top_k, top_p=top_p)
    else:
        y = generate(target_model, x, max_new_tokens, 
                temperature=temperature, top_k=top_k, top_p=top_p)
    
    # Synchronize CUDA if using GPU to get accurate timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate tokens generated (excluding the input tokens)
    num_tokens_generated = y.size(1) - x.size(1)
    tokens_per_second = num_tokens_generated / elapsed_time if elapsed_time > 0 else 0
    
    decoded = enc.decode(y[0,:].tolist())
    print("Generated text:\n"+ "-" * 100)
    print(decoded)
    print("-" * 100)
    if use_speculative:
        print(f"Acceptance ratio: {acceptance_ratio:.2%}")
    print(f"Tokens generated: {num_tokens_generated}")
    print(f"Time taken: {elapsed_time:.2f}s")
    print(f"Tokens/s: {tokens_per_second:.2f}")
    print("-" * 100)

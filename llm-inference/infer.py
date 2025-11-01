
import math
import time
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F
from model import GPT

# -------------------------------------------------------------------------#
# Input parameters
num_samples = 1 # number of samples to generate
# for greedy decoding keeps it 1 for now as all the samples are the same
max_new_tokens = 200 # maximum number of new tokens to generate
do_sample = True # Multinomial sampling (True) or greedy decoding (False)
temperature = 1.0 # temperature for sampling
top_k = 50 # top-k sampling (num. of highest prob vocab tokens to keep)
top_p = 0.9 # top-p sampling (cumulative probability threshold)
start_seq = "The following is a short story about a cat:" # start sequence
device = "cuda" # device to use
dtype = "float32" # "float16" or "bfloat16" or "float32"
use_cache = True # use KV cache
model_name = "gpt2-large" # model name
seed = 1337 # seed for the random number generator
presence_penalty = 0.0 # decreases likelihood of previously seen tokens
frequency_penalty = 0.0 # decreases likelihood proportionally to usage count
repetition_penalty = 1.0 # scales logits for seen tokens
# To penalize and reduce repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. 
# To reward and encourage repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.
# -------------------------------------------------------------------------#

# ---------------- Initialize the model ---------------- #
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

# load the model
model = GPT.from_pretrained(model_name)
print("Model loaded successfully")


# eval mode and move to appropriate device
model.eval()
model.to(device)

# ---------------- Initialize the tokenizer ---------------- #
enc = tiktoken.get_encoding("gpt2")

# ---------------- Encode the start sequence ---------------- #
tokens = enc.encode(start_seq, allowed_special={"<|endoftext|>"})  # n tokens (list of integers)
x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]  # (1, n)

# ---------------- Generate the text ---------------- #
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False,
             top_k=None, top_p=None, use_cache=True, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0):
    
    # handle temperature close to 0
    if temperature is None or math.isclose(temperature, 0.0):
        print("Warning: Temperature is close to 0, flip do_sample to False")
        do_sample = False
    
    # sanity checks
    if temperature is not None:
        assert temperature > 0.0, "Temperature must be greater than 0"
    if top_k is not None:
        assert isinstance(top_k, int) and top_k > 0, "top_k must be a positive integer"
    if top_p is not None:
        assert isinstance(top_p, float) and top_p > 0 and top_p <= 1, "top_p must be a float between 0 and 1"
    if top_p is not None:
        assert isinstance(top_p, float), "top_p must be a float"
    assert isinstance(presence_penalty, float) and 0.0 <= presence_penalty <= 1.0, "presence_penalty must be a float between 0 and 1"
    assert isinstance(frequency_penalty, float) and frequency_penalty >= 0.0 and frequency_penalty <= 1.0, "frequency_penalty must be a float between 0 and 1"
    assert isinstance(repetition_penalty, float) and repetition_penalty > 0.0, "repetition_penalty must be a float and greater than 0, below <1.0 means penalize repeats, >1.0 means penalize non-repeats"
    
    # clear KV cache before starting generation
    if use_cache:
        model.clear_kv_cache()
    
    for i in range(max_new_tokens):
        # With KV cache: only pass new tokens after prefill
        if use_cache and i > 0:
            # Only pass the last token (just generated)
            idx_cond = idx[:, -1:]
        else:
            # First pass (prefill) or no cache: pass full context (cropped if needed)
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
    
        # forward the model to get the logits
        logits, _ = model(idx_cond, use_cache=use_cache)  # (B,T,vocab_size) idx_cond: (B,T)
    
        # logits at last position
        logits = logits[:, -1, :]  # (B, vocab_size)
    
        # apply penalties before sampling to discourage repeats
        if presence_penalty > 0.0 or frequency_penalty > 0.0 or repetition_penalty != 1.0:
            # count the number of times each token has appeared in the sequence
            token_counts = torch.zeros_like(logits)
            # scatter_add_ adds the values in the second tensor to self using the indices in the first tensor
            token_counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=logits.dtype))
            # apply presence penalty
            if presence_penalty > 0.0:
                logits = logits - presence_penalty * (token_counts > 0).float()
            # apply frequency penalty
            if frequency_penalty > 0.0:
                logits = logits - frequency_penalty * token_counts
            # apply repetition penalty
            if repetition_penalty != 1.0:
                seen_mask = token_counts > 0
                penalty = torch.full_like(logits, repetition_penalty)
                positive_mask = seen_mask & (logits > 0)
                negative_mask = seen_mask & (logits < 0)
                logits = torch.where(positive_mask, logits / penalty, logits)
                logits = torch.where(negative_mask, logits * penalty, logits)

        # sample from the distribution or greedy decoding
        if do_sample:
            # scale the logits to the temperature
            if temperature < 0.1:
                # shift the logits to [-inf, 1] range for numerical stability
                logits = logits - logits.max(dim=-1, keepdim=True).values + 1
            logits = logits / temperature
    
            # top-k sampling
            if top_k is not None and top_k > 0:
                # select the top-k tokens 
                topk_probs, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                # set all tokens outside the top-k to -inf, thus softmax will set them to 0
                # topk_probs[:, [-1]] -> (B, 1) instead of topk_probs[:, -1] -> (B,)
                # topk_probs[:, [-1]] ensures each row of logits is compared elementwise to its own threshold value (topk_probs[i, -1]).
                logits[logits < topk_probs[:, [-1]]] = -float('inf')
    
            # top-p sampling
            if top_p is not None and top_p > 0:
                # sort the logits
                sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                # calculate the cumulative probabilities
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # create a mask of tokens to remove based on top-p
                sorted_indices_to_remove = cum_probs > top_p
                # keep the first token always, similar to huggingface implementation in https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
                # this is different from the original nucleus sampling implementation where the token that crosses the threshold is included
                sorted_indices_to_remove[...,[0]] = False
                # set all tokens outside the top-p to -inf, thus softmax will set them to 0
                sorted_logits[sorted_indices_to_remove] = -float('inf')
                # scatter back to original order using inverse permutation
                logits = torch.scatter(logits, -1, sorted_indices, sorted_logits)
    
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        else:
            # greedy decoding: select the token with the highest probability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
    
        # append to the sequence
        idx = torch.cat([idx, idx_next], dim=1)
    
        # early stopping if the token is the EOS token
        if idx_next == model.config.eos_token_id:
            print("EOS token encountered, stopping generation")
            break
    return idx

# print the generated text
print("Generated text:\n"+ "-" * 100)
with ctx:
    for _ in range(num_samples):
        start_time = time.time()
        y = generate(model, x, max_new_tokens, 
                    temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p, use_cache=use_cache,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty)
        
        # Synchronize CUDA if using GPU to get accurate timing
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate tokens generated (excluding the input tokens)
        num_tokens_generated = y.size(1) - x.size(1)
        tokens_per_second = num_tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        decoded = enc.decode(y[0,:].tolist())
        print(decoded)
        print("-" * 100)
        print(f"Tokens generated: {num_tokens_generated}")
        print(f"Time taken: {elapsed_time:.2f}s")
        print(f"Tokens/s: {tokens_per_second:.2f}")
        print("-" * 100)

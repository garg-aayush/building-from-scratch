
import math
import time
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F
from model import GPT
from torch.nn.utils.rnn import pad_sequence

# -------------------------------------------------------------------------#
# Input parameters
num_samples = 1 # number of samples to generate
# for greedy decoding keeps it 1 for now as all the samples are the same
max_new_tokens = 200 # maximum number of new tokens to generate
do_sample = False # Multinomial sampling (True) or greedy decoding (False)
temperature = 1.0 # temperature for sampling
top_k = 50 # top-k sampling (num. of highest prob vocab tokens to keep)
top_p = 0.9 # top-p sampling (cumulative probability threshold)
start_seq = ["<|endoftext|>", "The following is a short story about a cat:", "What is the capital of France?"] # start sequence
device = "cuda" # device to use
dtype = "bfloat16" # "float16" or "bfloat16" or "float32"
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
# sanity check
assert len(start_seq) > 0, "start_seq must contain at least one prompt"
# convert the start sequence to a list if it is a string
start_sequences = [start_seq] if isinstance(start_seq, str) else list(start_seq)
# pad_token_id == eos_token_id for GPT-2
pad_token_id = model.config.eos_token_id

# encode the start sequences
def prepare_batch(start_sequences, pad_token_id, block_size, use_cache=False):
    token_tensors = []
    lengths = []
    for text in start_sequences:
        # encode the text using the GPT-2 tokenizer
        encoded = enc.encode(text, allowed_special={"<|endoftext|>"})
        # handle empty sequences by using pad token
        if len(encoded) == 0:
            encoded = [pad_token_id]
        # crop the encoded sequence to the block size
        elif len(encoded) > block_size:
            encoded = encoded[-block_size:]
        # convert the encoded sequence to a tensor
        encoded_tensor = torch.tensor(encoded, dtype=torch.long, device=device)
        token_tensors.append(encoded_tensor)
        lengths.append(encoded_tensor.size(0))
    
    # check if all sequences are of the same token length
    has_variable_lengths = len(set(lengths)) > 1
    if has_variable_lengths and use_cache:
        print(f"Warning: variable-length sequences detected (lengths: {lengths}). "
              f"Disabling KV cache (use_cache=False) as only uniform-length batches are supported with cache.")
        use_cache = False
    
    # pad the sequences to the same length (right padding for GPT-2 as it uses absolute position embedding)
    input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=pad_token_id, padding_side="right") # (B, T)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    attention_mask = torch.arange(input_ids.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return input_ids, attention_mask, lengths, use_cache

x, attention_mask, initial_lengths, use_cache = prepare_batch(start_sequences, pad_token_id, model.config.block_size, use_cache)
print("Attention mask:")
print(attention_mask)
print("\nInput IDs:")
print(x)
print("\nInitial lengths:")
print(initial_lengths)

# ---------------- Generate the text ---------------- #
@torch.no_grad()
def generate(model, idx, max_new_tokens, attention_mask, initial_lengths, temperature=1.0, do_sample=False,
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
    
    batch_size, _ = idx.size()
    pad_token_id = model.config.eos_token_id
    
    # if attention mask is not provided, create a mask of all True
    if attention_mask is None:
        attention_mask = torch.ones_like(idx, dtype=torch.bool, device=idx.device)
    seq_lengths = initial_lengths.clone()
    print("Sequence lengths: ", seq_lengths)
    print("Batch size: ", batch_size, "Pad token ID: ", pad_token_id)
    print("Attention mask: ", attention_mask)
    
    # create finished mask (for early stopping)
    # A sequence is finished if:
    # 1. It has length 0 (empty sequence), or
    # 2. Its last token is the EOS token
    finished = seq_lengths == 0
    if (~finished).any():
        # Gather the last token from each sequence using the last_positions
        last_tokens = idx.gather(1, torch.clamp(seq_lengths - 1, min=0).unsqueeze(1)).squeeze(1)
        # Mark sequence as finished if its last token is EOS
        finished = finished | (last_tokens == model.config.eos_token_id)
    
    batch_indices = torch.arange(batch_size, device=idx.device)
    print("Finished mask: ", finished)
    print("Batch indices: ", batch_indices)
    
    for i in range(max_new_tokens):
        # Early stopping if all sequences are finished
        if finished.all():
            break
        
        # With KV cache: only pass new tokens after prefill
        if use_cache and i > 0:
            # Only pass the last token (just generated)
            idx_cond = idx[:, -1:]
        else:
            # First pass (prefill) or no cache: pass full context (cropped if needed)
            if idx.size(1) <= model.config.block_size:
                idx_cond = idx
                effective_lengths = seq_lengths.clone()
            else:
                idx_cond = idx[:, -model.config.block_size:]
                effective_lengths = torch.clamp(seq_lengths, max=model.config.block_size)
        
        # forward the model to get the logits
        logits, _ = model(idx_cond, use_cache=use_cache, attn_mask_padding=attention_mask)  # (B,T,vocab_size) idx_cond: (B,T), attention_mask: (B, T)
    
        # logits at last position
        if use_cache and i > 0:
            logits = logits[:, -1, :]
        else:
            last_positions = effective_lengths - 1
            # (B,) -view-> (B, 1) -expand-> (B, 1, vocab_size) -gather-> (B, 1, vocab_size) -squeeze-> (B, vocab_size)
            logits = logits.gather(1, last_positions.view(batch_size, 1, 1).expand(-1, 1, logits.size(-1))).squeeze(1)  # (B, vocab_size)
    
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
    
        # check if any sequence is finished
        if finished.any():
            # torch.where(condition, x, y) -> returns x if condition is True, y if condition is False
            # finished.view(-1, 1) -> (B, 1)
            idx_next = torch.where(finished.view(-1, 1), torch.full_like(idx_next, pad_token_id), idx_next)
        # create active sequence mask
        active_mask = ~finished
        
        # append to the sequence
        max_req_idx = seq_lengths[active_mask].max()
        if idx.size(1) <= max_req_idx:
            # expand the sequence by adding a column of padding tokens
            idx = torch.cat([idx, torch.full((batch_size, 1), pad_token_id, dtype=idx.dtype, device=idx.device)], dim=1)
            # expand the attention mask by adding a column of False values
            attention_mask = torch.cat([attention_mask, torch.zeros((batch_size, 1), dtype=torch.bool, device=idx.device)], dim=1)
        
        # update idx with the new token and attention mask
        pos = seq_lengths[active_mask]
        idx[batch_indices[active_mask], pos] = idx_next[active_mask, 0]
        attention_mask[batch_indices[active_mask], pos] = True
        seq_lengths[active_mask] += 1
        
        # update finished mask
        finished = finished | ((idx_next.squeeze(1) == model.config.eos_token_id) & active_mask)

    return idx, seq_lengths

# print the generated text
print("Generated text:\n"+ "-" * 100)
with ctx:
    for _ in range(num_samples):
        start_time = time.time()
        y, final_lengths = generate(model, x, max_new_tokens, attention_mask, initial_lengths,
                    temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p, use_cache=use_cache,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty)
        
        # Synchronize CUDA if using GPU to get accurate timing
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate tokens generated (excluding the input tokens)
        tokens_generated = (final_lengths - initial_lengths).cpu()
        print("Final lengths: ", final_lengths)
        print("Initial lengths: ", initial_lengths)
        print("Tokens generated: ", tokens_generated)
        total_tokens_generated = tokens_generated.clamp_min(0).sum().item()
        tokens_per_second = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        for sample_idx, length in enumerate(final_lengths.tolist()):
            decoded = enc.decode(y[sample_idx,:length].tolist())
            print("-" * 100)
            print(f"Sample {sample_idx} tokens generated: {tokens_generated[sample_idx].item()}")
            print("-" * 100)
            print(decoded)

        print(f"Time taken: {elapsed_time:.2f}s")
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Tokens/s: {tokens_per_second:.2f}")
        print("-" * 100)

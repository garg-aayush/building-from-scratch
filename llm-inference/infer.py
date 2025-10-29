
import math

import tiktoken
import torch
import torch.nn.functional as F
from model import GPT

# -------------------------------------------------------------------------#
# Input parameters
num_samples = 1 # number of samples to generate
# for greedy decoding keeps it 1 for now as all the samples are the same
max_new_tokens = 50 # maximum number of new tokens to generate
do_sample = True # Multinomial sampling (True) or greedy decoding (False)
<<<<<<< HEAD
temperature = 1.0 # temperature for sampling
=======
>>>>>>> d141d2f (Basic inference imp: add multinomial sampling)
start_seq = "The following is a short story about a cat:" # start sequence
device = "cpu" # device to use
model_name = "gpt2-large" # model name
seed = 1337 # seed for the random number generator
# -------------------------------------------------------------------------#

# ---------------- Initialize the model ---------------- #
# set the seed
torch.manual_seed(seed)
if device == "cuda": 
    torch.cuda.manual_seed(seed)

# load the model
model = GPT.from_pretrained(model_name)
print("Model loaded successfully")

# available device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

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
<<<<<<< HEAD
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False):
    # handle temperature close to 0
    if temperature is None or math.isclose(temperature, 0.0):
        print("Warning: Temperature is close to 0, flip do_sample to False")
        do_sample = False
=======
def generate(model, idx, max_new_tokens, do_sample=False):
>>>>>>> d141d2f (Basic inference imp: add multinomial sampling)
    for _ in range(max_new_tokens):
        # crop the context if it's greater than the block size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits
        logits, _ = model(idx_cond)  # (B,T,vocab_size) idx_cond: (B,T)
        # logits at last position
        logits = logits[:, -1, :]  # (B, vocab_size)
<<<<<<< HEAD
        # sample from the distribution or greedy decoding
        if do_sample:
            # scale the logits to the temperature
            if temperature < 0.1:
                # shift the logits to [-inf, 1] range for numerical stability
                logits = logits - logits.max(dim=-1, keepdim=True).values + 1
            logits = logits / temperature
            # sample from the distribution (multinomial sampling)
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
=======
        if do_sample:
            # sample from the distribution (multinomial sampling)
            probs = F.softmax(logits, dim=-1)
>>>>>>> d141d2f (Basic inference imp: add multinomial sampling)
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
print("Generated text:\n")
for _ in range(num_samples):
<<<<<<< HEAD
    y = generate(model, x, max_new_tokens, temperature, do_sample)
=======
    y = generate(model, x, max_new_tokens, do_sample)
>>>>>>> d141d2f (Basic inference imp: add multinomial sampling)
    decoded = enc.decode(y[0,:].tolist())
    print(decoded)
    print("-" * 100)

import tiktoken
import torch
from model import GPT

# -------------------------------------------------------------------------#
# Input parameters
num_samples = 1 # number of samples to generate
# for greedy decoding keeps it 1 for now as all the samples are the same
max_new_tokens = 10 # maximum number of new tokens to generate
start_seq = "Hello, I'm a language model," # start sequence
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
tokens = enc.encode(start_seq)  # n tokens (list of integers)
x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]  # (1, n)

# ---------------- Generate the text ---------------- #
@torch.no_grad()
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # forward the model to get the logits
        logits, _ = model(idx)  # (B,T,vocab_size)
        # logits at last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # greedy decoding: select the token with the highest probability
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
        # append to the sequence
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

# print the generated text
print("Generated text:\n")
for _ in range(num_samples):
    y = generate(model, x, max_new_tokens)
    decoded = enc.decode(y[0,:].tolist())
    print(decoded)
    print("-" * 100)
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 9000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
# check if device is available ('cuda' or 'mps' or 'cpu')
device = "cpu"  # default fallback
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
device = "cpu"
# ------------------

print(f"Using {device} device")
torch.manual_seed(42)

# !mkdir -p data && wget -O data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# read the dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
# char to int mapping
stoi = {ch: i for i, ch in enumerate(chars)}
# int to char mapping
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: maps the given string to a list of int (tokens)
encode = lambda s: [stoi[c] for c in s]
# decoder: maps the list of int (tokens) to string
decode = lambda l: "".join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# context manager to disable gradient calculation, pytorch can be more efficient with memory as it doesn't need to store the gradients
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(
            eval_iters
        ):  # calculate loss eval_iters times so we can get a better estimate
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    data = train_data if split == "train" else val_data
    # get batch_size random indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocal_size):
        super().__init__()
        # lookup embedding table where we have an embedding row each token in vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs, targets: B (batch) X T (block_size)
        logits = self.token_embedding_table(inputs)  # (B, T, C) C: embedding_length

        if targets is None:
            loss = None
        else:
            # targets: (B,T)
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)  # expected value: -ln(1/65)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # inputs: (B,T)
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(inputs)
            # focus only on last token
            logits = logits[:, -1, :]  # (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the sequence
            inputs = torch.cat((inputs, idx_next), dim=1)  # (B,T+1)
        return inputs


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for steps in range(max_iters):

    # evaluate the loss
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step: {steps} -> loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    # set_to_none=True is more memory efficient than setting to zero
    # as it deallocates the gradient tensors instead of filling them with zeros
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

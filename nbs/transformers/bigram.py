import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # number of sequences processed in parallel
block_size = 8  # maximum context length for predictions
max_iters = 9000  # total number of training iterations
eval_interval = 300  # evaluate model every N steps
learning_rate = 1e-3  # learning rate for optimizer
eval_iters = 200  # number of iterations to average for loss estimation
# check if device is available ('cuda' or 'mps' or 'cpu')
device = "cpu"  # default fallback
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
# ------------------

print(f"Using {device} device")
torch.manual_seed(42)

# !mkdir -p data && wget -O data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# read the dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
# create character-to-integer mapping
stoi = {ch: i for i, ch in enumerate(chars)}
# create integer-to-character mapping
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: converts string to list of integers (tokenization)
encode = lambda s: [stoi[c] for c in s]
# decoder: converts list of integers back to string (detokenization)
decode = lambda l: "".join([itos[i] for i in l])


# Split data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% for training
train_data = data[:n]
val_data = data[n:]


# Disable gradient calculation for evaluation - more memory efficient
@torch.no_grad()
def estimate_loss():
    """Estimate average loss over multiple batches for both train and validation sets"""
    out = {}
    model.eval()  # switch to evaluation mode
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):  # average over multiple batches for better estimate
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # switch back to training mode
    return out


def get_batch(split):
    """Generate a batch of input-target pairs from the specified split"""
    data = train_data if split == "train" else val_data
    # randomly sample batch_size starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # create input sequences of length block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    # create target sequences (shifted by one position)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocal_size):
        super().__init__()
        # embedding table: each token gets a learned vector representation
        # for bigram model, embedding size equals vocab_size to directly predict next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs: (batch_size, sequence_length)
        logits = self.token_embedding_table(
            inputs
        )  # (batch_size, sequence_length, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape for cross-entropy loss calculation
            logits = rearrange(
                logits, "b t c -> (b t) c"
            )  # flatten batch and time dimensions
            targets = rearrange(targets, "b t -> (b t)")  # flatten targets
            loss = F.cross_entropy(logits, targets)  # compute cross-entropy loss

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        """Generate new tokens by sampling from the model's predictions"""
        # inputs: (batch_size, sequence_length)
        for _ in range(max_new_tokens):
            # get predictions for current sequence
            logits, _ = self(inputs)
            # focus only on the last time step's predictions
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            # convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            # append sampled token to the sequence
            inputs = torch.cat(
                (inputs, idx_next), dim=1
            )  # (batch_size, sequence_length + 1)
        return inputs


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create optimizer for model parameters
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# main training loop
for steps in range(max_iters):

    # periodically evaluate model performance
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step: {steps} -> loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}"
        )

    # get a batch of training data
    xb, yb = get_batch("train")

    # forward pass and loss calculation
    logits, loss = m(xb, yb)
    # clear gradients (set_to_none=True is more memory efficient)
    optimizer.zero_grad(set_to_none=True)
    # backward pass to compute gradients
    loss.backward()
    # update model parameters
    optimizer.step()

# generate text from the trained model
context = torch.zeros(
    (1, 1), dtype=torch.long, device=device
)  # start with empty context
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

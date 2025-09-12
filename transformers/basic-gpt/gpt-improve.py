import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # number of sequences processed in parallel
block_size = 256  # maximum context length for predictions
max_iters = 5000  # total number of training iterations
eval_interval = 50  # evaluate model every N steps
learning_rate = 3e-4  # learning rate for optimizer
eval_iters = 200  # number of iterations to average for loss estimation
n_embed = 384  # number of embedding dimensions
n_layer = 6  # number of Transformer blocks
n_head = 6  # number of attention heads
dropout = 0.2  # dropout rate
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
with open("./data/input.txt", "r", encoding="utf-8") as f:
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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # k,q,v projections for all heads in parallel (B,T,n_embed)
        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        # projection layer: allows the model to learn how to combine information from multiple attention heads into a unified representation for the residual pathway
        self.proj = nn.Linear(n_embed, n_embed)
        # dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)
        self.heads = num_heads
        self.head_size = head_size
        # we use register_buffer to create a tensor that is not a parameter of the model but is part of the model's state and will be saved and loaded with the model
        # we view it as (1, 1, block_size, block_size) to make it compatible with the shape of the attention scores
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        # output size: (batch_size, sequence_length, num_heads * head_size = n_embed)
        B, T, C = x.shape  # (B,T,n_embed)
        # calculate query, key, values for all heads
        k = self.key(x)  # (B,T,n_embed)
        q = self.query(x)  # (B,T,n_embed)
        v = self.value(x)  # (B,T,n_embed)
        # rearrange for all heads in parallel
        k = rearrange(k, "B T (num_heads head_size) -> B num_heads T head_size", num_heads=self.heads)
        q = rearrange(q, "B T (num_heads head_size) -> B num_heads T head_size", num_heads=self.heads)
        v = rearrange(v, "B T (num_heads head_size) -> B num_heads T head_size", num_heads=self.heads)
        # compute attention scores ("affinities")
        w = einsum(q, k, "... T1 head_size, ... T2 head_size -> ... T1 T2")  # (B,num_heads,T,T)
        w = w.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))  # (B,num_heads,T,T)
        w = F.softmax(w, dim=-1)  # (B,num_heads,T,T)
        w = self.attn_dropout(w)
        # perform weighted aggregation of the values
        out = w @ v  # (B,num_heads,T,T) X (B,num_heads,T,head_size) -> (B,num_heads,T,head_size)
        out = rearrange(out, "B n_heads T head_size -> B T (n_heads head_size)")  # (B,T,n_embed)
        # perform weighted aggregation of the values
        out = self.res_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # 4* to match the Transformer paper
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(), # instead of ReLU
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication (MHA) followed by computation (FFN)"""

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # pre-layer normalization
        x = x + self.sa(self.ln1(x))  # residual connection
        x = x + self.ffwd(self.ln2(x))  # residual connection
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # embedding table: each token gets a learned vector representation
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # used to get logits from the embeddings
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # positional embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)

    def forward(self, inputs, targets=None):
        # inputs: (batch_size, sequence_length)
        B, T = inputs.shape
        # (batch_size, sequence_length, n_embed)
        tok_embd = self.token_embedding_table(inputs)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd  # (batch_size, sequence_length, n_embed)
        x = self.blocks(x)  # (batch_size, sequence_length, n_embed)
        x = self.ln_f(x)  # (batch_size, sequence_length, n_embed)
        logits = self.lm_head(x)  # (batch_size, sequence_length, vocab_size)

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
            # crop context if it's too long (as the position embeddings will run out of scope)
            inputs_cond = inputs[:, -block_size:]
            # get predictions for current sequence
            logits, _ = self(inputs_cond)
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


model = BigramLanguageModel()
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

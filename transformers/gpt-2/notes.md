# Let's reproduce GPT-2 (124M)

## Links
- [Youtube Video Link](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1388s)

## Introduction
- In this video, we reproduce the GPT-2 model, specifically the 124M parameter version.
- OpenAI released GPT-2 in 2019, along with a blog post, [research paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), and the original [TensorFlow code](https://github.com/openai/gpt-2) on GitHub.
- GPT-2 isn’t a single model but a *mini-series* of models of different sizes. The smallest has **124M** parameters, and the largest has **1.5B**. The reason for these variants is so researchers can analyze scaling laws—how model performance on tasks like translation, summarization, and Q\&A improves with size.
<img src="images/gpt-2-sizes.png" alt="gpt-2 sizes" width="400">
- GPT-2 (124M):
    * It has 12 Transformer layers.
    * Each layer has a hidden size of 768.

- If we reproduce this correctly, by the end of training we should achieve validation losses similar to the original model (good at next-token prediction on unseen validation data).

- Note, GPT-2 paper omits training details, so we’ll also reference the GPT-3 paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), which is more concrete about hyperparameters and optimizers.


---

## Loading the Original GPT-2 (Huggingface version)

- The original GPT-2 implementation was in TensorFlow, but we’ll use Hugging Face’s PyTorch implementation, which is easier to work with. 
- Using Hugging Face, `gpt2` corresponds to the 124M model, while larger ones use `gpt2-medium`, `gpt2-large`, and `gpt2-xl`.
- GPT-2's vocabulary has 50,257 tokens, each mapped to a 768-dimensional embedding vector. It uses learned positional embeddings of size: 768.
    - Unlike the original Transformer paper where sinusoidal embeddings were fixed, GPT-2 trains these parameters from scratch and they converge to similar patterns (sine-cosine patterns)
    - Here channels show noisy curves, suggesting the model wasn't fully trained to convergence. However, 
- Please see the [load-hf-gpt2.ipynb](load-hf-gpt2.ipynb) for more details.

---

## Building GPT-2 model class

- Instead of relying on Hugging Face’s \~2000 lines of code, we’ll write a minimal implementation (\~100 lines) with the same module/parameter structure so we can import GPT-2 weights. 
- Key architectural points difference from original _Attention_ paper :
    * Decoder-only Transformer (no encoder, no cross-attention).
    * LayerNorms moved before Attention/MLP (pre-norm).
        - Pre-norm ensures residual connections form a clean pathway from inputs to outputs, allowing gradients to flow effectively as `+` operation distributes the gradient equally across the two pathways. It is good for optimization. 
    * Extra LayerNorm before the LM head.
- You can think attention is the “reduce” operation, where tokens exchange information by attending to others, while the MLP is the “map” operation, applied independently to each token. Thus, each block can be seen as a sequence of map-reduce steps, progressively refining token representations.
- The MLP in GPT-2 uses the [GELU](https://arxiv.org/abs/1606.08415) activation function, specifically the approximate version based on `tanh`. 
    - This choice is historical—at the time, the exact GELU implementation was slower in TensorFlow, so the approximation was used and carried over into BERT and GPT-2. See this [blog](https://github.com/pytorch/pytorch/issues/39853).
    - While today we’d prefer the exact version, we’ll reproduce GPT-2 faithfully by using the approximation. 
    - GELU has advantages over ReLU since it avoids the “dead relu neuron” problem and provides smoother gradients.
        -  If a ReLU neuron is exactly flat at zero, any activations that fall there will get exactly zero gradient. There's no change, no adaptation, no development of the network.
        - The GELU always contributes a local gradient and so there's always going to be a change always going to be an adaptation and sort of smoothing it out ends up empirically working better in practice
- Multi-head attention is implemented with tensor reshaping (4D-Tensor) for efficient parallel computation.

---

## Forward Pass & Sampling

- The forward pass takes input token indices (B,T) and outputs logits (B,T,vocab_size). Applying softmax gives next-token probabilities. This is the T+1 logits.

- Sampling:
    * We start with a prefix (e.g., *“Hello, I’m a language model”*).
    * Take the model’s output logits. Apply softmax to get probabilities. This is the T+1 logits.
    * Repeatedly sample from logits using **top-k sampling (k=50)** (default in Hugging Face pipeline).
    * Top-k sampling restricts sampling to the top-k most probable tokens. This improves coherence by avoiding low-probability words that derail the text. If we sample from the full distribution, completions tend to go off track; restricting to top-k helps a lot. In top-k, it is done by creating a mask of the top-k most probable tokens and setting the rest to 0. Then, we sample from the masked renormalized distribution.
    * Append sampled tokens until max length is reached.
    
- The implementation generates coherent text, though Hugging Face’s pipeline has slight differences due to additional sampling heuristics.

---

## Dataset Choice: Tiny Shakespeare

- For debugging and quick iteration, we don’t start with a huge dataset. Instead, we use the **Tiny Shakespeare dataset**:
    * It’s small enough to train on a laptop/GPU quickly, but large enough to test whether the model learns.
    * Text is plain ASCII, so each character maps to a single byte.
- This dataset allows us to debug the training pipeline before moving to larger corpora.
- Here,
    * **Inputs (X):** the current sequence of tokens.
    * **Targets (Y):** the same sequence shifted left by one position.
---

## Loss function: cross-entropy loss

- We use **cross-entropy loss**, the standard for classification problems:
    * The logits from the model are of shape `(batch_size, sequence_length, vocab_size)`.
    * Targets are `(batch_size, sequence_length)`.
    * Cross-entropy flattens these and computes how well the predicted distribution matches the actual next token.

- At initialization, the model’s predictions are close to uniform across the 50,257 tokens.
    * Expected loss at this stage = `-ln(1/vocab_size)` ≈ **10.8**. This is the loss we expect when the model is making random predictions. This shows the model is not favoring any one token over the others. The probability of each token is fairly diffused.
    * If we see a loss near this, we know our pipeline is wired correctly. 
---

## Optimize on single set of batch
- We train using **AdamW**, a variant of the Adam optimizer that fixes weight decay behavior:
    * Make sure to start by zeroing gradients (`optimizer.zero_grad()`).
- Before training on the whole dataset, we first **train repeatedly on one tiny batch** as a sanity check.
    * If the model can drive loss close to zero, it means our implementation of forward/backward/optimizer is correct. This shows the model is learning to overfit the small batch, which is a standard sanity check in deep learning.
    * If it fails to overfit, there's a bug in the pipeline.
---

## Minimal dataloader
Next I implement a minimal **data loader**:
- Read the whole tokenized dataset once and maintain a current position
- On each `next_batch()` call:
    * Return contiguous chunks of size `B*T+1` (the +1 is for creating labels)
    * Advance position by `B*T` tokens
    * Wrap back to the start when past the end of dataset
- Running only 50 steps shows loss dropping from ~11 to ~6.6:
    * This is mostly from learning that many vocab entries (unicodes) never occur (driving their logits down). 
    * Plus some early learning signal
    * Loss won't reach zero without training for full epochs
---

## Fix the bug relative to GPT-2

There's a subtle **bug** relative to GPT-2: 
- Until now, we didn't consider the **weight tying** between input embeddings and the output classifier layers
- In GPT-2 ([Attention Is All You Need](https://arxiv.org/abs/1706.03762) following [Press & Wolf 2017](https://arxiv.org/pdf/1608.05859)), the token embedding matrix `wte.weight` and the classifier matrix used before the final softmax share **the same weights**. You can check this by comparing the state dicts `wte.weight` and `lm_head.weight`. They should be equal and point to same data pointer
- Tying has two benefits:
    * (1) It enforces that token similarity is consistent between embedding space and output distribution
    * (2) It saves parameters—here `768 × 50257 ≈ 40M` weights, \~30% of the 124M model
---

## Initialization Matching GPT-2

- We need to ensure my initialization matches GPT-2's approach:
- Initialize embedding and linear layers:
    * **Linear weights:** Initialize with `Normal(0, 0.02)` distribution
    * **Biases:** Set to 0 
    * **Embeddings:** Use `Normal(0, 0.02)`
    * **LayerNorm:** Use PyTorch defaults (`weight=1`, `bias=0`)
- GPT-2 applies special scaling to residual branch weights at initialization:
    * Scale certain residual-branch weights by `1/√N`, where `N` = number of residual additions along the depth
    * Each block has 2 residual additions (attention + MLP), so `N = 2 * n_layer`
    * **Intuition:** The residual stream repeatedly does `x ← x + contribution`. Without scaling, variance of `x` grows like `sqrt(N)`. Scaling by `1/sqrt(N)` keeps forward activations controlled.
---
# Rotary Positional Embeddings (RoPE) Notes

## Resources:
- [Llama explained video by Umar Jamil](https://www.youtube.com/watch?v=Mn_9W1nCFLo): Refer to RoPE section starting at 24:30 timestamp 
- [Efficient NLP's RoPE explanation](https://www.youtube.com/watch?v=o29P0Kpobz0)
- [RoPE paper](https://arxiv.org/abs/2104.09864)
- [Transformer's RoPE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L73)
- [RoPE implementation notebook](./play-nbs/rope.ipynb)

## What are positional embeddings (and why do we need them)?

Transformers don’t know token order by default (they are position invariant). If you feed them the same bag of words in a different order, they’d treat it the same unless **you inject position information**.

* **Positional embeddings** are vectors that encode **where** each token sits in the sequence.
* They let attention care about “who’s near whom” so “the dog chased the pig” ≠ “the pig chased the dog”.

## Absolute vs. Relative positional embeddings

### Absolute positional embeddings

* **Idea:** give each position $m$ its own vector $p_m$ and **add** it to the token embedding $x_m$:

  $$
  \tilde{x}_m = x_m + p_m.
  $$
* They can be either learned or fixed, as in the case of sine-cosine embeddings from the original Transformer paper.
* **Pros:** trivial to implement; works well; cache-friendly at inference.
* **Cons:** Poor extrapolation to sequences longer than those seen during training. It doesn't inherently model the relative positions or pairwise distances between tokens, as each position is treated independently.

### Relative positional embeddings

* **Idea:** modify the key vector with a bias $a_{\Delta}$ that depends on the relative distance $\Delta = n-m$.
* the formula for the attention score is given by:
  $$
  \mathrm{score}(m,n) = \frac{\langle Q_m, K_n + a_{\Delta}\rangle }{\sqrt{d_{\text{head}}}}
  $$
* **Pros:** The model “feels” how far tokens are, as pairwise distance information is added to the attention score.
* **Cons:** Can be more complex to implement and may not be as cache-friendly during inference. The relative positional information needs to be incorporated at each attention calculation, which can slow down inference for long sequences.


## What is RoPE (Rotary Positional Embeddings)?

### Intuition for RoPE
- The attention mechanism relies on the dot product, a form of inner product, to compute similarity scores between queries and keys.
- The core idea is to define an inner product between query ($q$) and key ($k$) vectors that only depends on the vectors themselves and the relative distance between their corresponding tokens.
- RoPE uses a neat trick: instead of **adding** position, you **rotate** each query/key vector by a **position-dependent angle** before the dot-product.
- RoPE groups the head dimension into many such 2-D pairs and rotates each pair by a position-dependent angle $\theta_{m,i}$. Do this for **queries** and **keys**:
    * Token at position $m$ → rotate by $\theta_{m,i}$.
    * Token at position $n$ → rotate by $\theta_{n,i}$.
- When you take the dot-product to form attention scores, those rotations **collapse** into a function of the **difference** $(n - m)$. So attention learns about **relative** positions naturally, with no extra pairwise tables.
* The attention score ends up depending on **relative offsets** $n-m$ even though you only did per-token operations.
- RoPE blends the good parts of both worlds: distance awareness like relative methods, but with a simple, cache-friendly implementation like absolute methods.

## Mathematical formulation (the essentials)
* Let $Q_m, K_n \in \mathbb{R}^{d_{\text{head}}}$ be per-token, per-head projections. Define $\tilde{Q}_m = R_m Q_m$ and $\tilde{K}_n = R_n K_n$, where $R_m$ is the block-diagonal rotation built from the blocks above. Attention scores per head:
$$
\mathrm{score}(m,n) = \frac{\langle \tilde{Q}_m, \tilde{K}_n\rangle}{\sqrt{d_{\text{head}}}}
= \frac{\langle Q_m, R_{n-m} K_n\rangle}{\sqrt{d_{\text{head}}}}.
$$
* Key point: the inner product depends on **$n-m$** (relative offset), not just $m$ or $n$.
* **Rotation blocks and frequencies**
  * Split a head of size $d_{\text{head}}$ into $d_{\text{head}}/2$ two-dimensional blocks. For block $i$:
    $$
    \theta_{m,i} = m\cdot \omega_i,\qquad 
    \omega_i = b^{-2i/d_{\text{head}}}
    $$
  * with base $b$ (commonly $10,000$). Lower-index blocks rotate **faster**, higher-index **slower**. Per block:

    $$
    R_{\theta_{m,i}}=\begin{bmatrix}
    \cos\theta_{m,i} & -\sin\theta_{m,i}\\
    \sin\theta_{m,i} & \phantom{-}\cos\theta_{m,i}
    \end{bmatrix}.
    $$

* It's applied to **Q and K only** (never V). RoPE only rotates the query and key vectors, not the value vectors. The rotation is **frequency-scaled** across dimensions (fast rotations in some dims, slow in others).

## Key takeaways (TL;DR)

* Transformers need order; absolute vs. relative give different inductive biases.
* **RoPE rotates Q/K** by position so attention naturally depends on **relative offsets**.
* It’s **small, fast, and cache-friendly**, with a built-in tendency to favor nearby tokens (yet allow long-range links).

* **KV cache friendly**: Earlier tokens don’t need recomputation when new tokens arrive; their rotations are fixed by their positions.


* Relative-position behavior without complex pairwise tables.
* Tiny implementation footprint; fast and cache-friendly.
* A gentle **locality bias** (distance-decay upper bound) that often helps.

**What to watch for**

* **Head dimension must be even** (we rotate in 2-D pairs).
* The frequency schedule matters; poor choices can hurt long-range generalization.
* Base RoPE is not a magical solution for infinite context. For very long contexts, consider frequency scaling/curves tailored to your window.

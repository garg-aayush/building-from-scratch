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

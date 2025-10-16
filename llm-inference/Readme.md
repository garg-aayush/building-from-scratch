# LLM Inference

This folder contains the code for running LLM inference. I'll build up the implementation incrementally through commits, adding incremental features and complexity one at a time. Below is the list I hope to implement 🤞.

## To Do

- [ ] Basic Inference Implementation
    - Basic greedy decoding (argmax) with EOS / max tokens.
    - Multinomial sampling + temperature scaling.
 
- [ ] Sampling Strategies I: Top-K and Top-P (nucleus) sampling
- [ ] Add similar guards as in Transformers library: Add repetition penalty, no-repeat-n-gram, stop strings

- [ ] Sampling Strategies II: Beam search comparison (show degeneration)

- [ ] Inference Speed Optimization I: Implement KV cache toggle; benchmark speedup

- [ ] Inference Speed Optimization II: Draft-verify speculative decoding (Try to implement)
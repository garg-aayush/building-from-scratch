# LLM Inference

This folder contains the code for running LLM inference. I'll 
build up the implementation incrementally through commits, 
adding incremental features and complexity one at a time. 
Below is the list I hope to implement 🤞.

## Implementation Progress

My starting point is `model.py` from GPT-2 work (commit: `a100995` on branch gpt2) and I will be building inference logic in `infer.py`.

## To Do
- [-] Basic Inference Implementation
    - [x] **Greedy Decoding**: Implement argmax-based token selection with max_new_tokens parameter
    - [x] **EOS Token Handling**: Stop generation when EOS token is encountered
    - [ ] **Context Window Management**: Implement sliding window to keep sequence length ≤ block_size
    - [ ] **Multinomial Sampling**: Add basic sampling option (sample from full distribution)
    - [ ] **Temperature Scaling**: Add temperature parameter to control sampling randomness

- [ ] Sampling Strategies I: Top-K and Top-P (nucleus) sampling
- [ ] Add similar guards as in Transformers library: Add repetition penalty, 
no-repeat-n-gram, stop strings

- [ ] Sampling Strategies II: Beam search comparison (show degeneration)

- [ ] Inference Speed Optimization I: Implement KV cache toggle; benchmark speedup

- [ ] Inference Speed Optimization II: Draft-verify speculative decoding (Try to 
implement)
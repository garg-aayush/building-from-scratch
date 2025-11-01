# LLM Inference

This folder contains the code for running LLM inference. I'll 
build up the implementation incrementally through commits, 
adding incremental features and complexity one at a time. 
Below is the list I hope to implement 🤞.

## Implementation Progress

My starting point is `model.py` from GPT-2 work (commit: `a100995` on branch gpt2) and I will be building inference logic in `infer.py`.

## To Do
- [x] Basic Inference Implementation
    - [x] **Greedy Decoding**: Implement argmax-based token selection with max_new_tokens parameter
    - [x] **EOS Token Handling**: Stop generation when EOS token is encountered
    - [x] **Context Window Management**: Implement sliding window to keep sequence length ≤ block_size
    - [x] **Multinomial Sampling**: Add basic sampling option (sample from full distribution)
    - [x] **Temperature Scaling**: Add temperature parameter to control sampling randomness

- [x] Sampling Strategies I: Top-K and Top-P (nucleus) sampling
    - [x] **Top-K Sampling**: Implement top-k filtering (keep only k most likely tokens)
    - [x] **Top-P (Nucleus) Sampling**: Implement nucleus sampling (cumulative probability threshold). Support both top-k and top-p sampling together.
    - [x] **Sanity Checks**: Add sanity checks for temperature, top-k, and top-p.

- [x] Add Penalty Controls
    - [x] **Presence Penalty**: Adjust logits when a token has already appeared
    - [x] **Frequency Penalty**: Scale down logits proportionally to repetition counts
    - [x] **Repetition Penalty**: Apply configurable penalty factor akin to Transformers
    > Note: I have implemented all three penalties in `infer.py`. However, tbh I don't think they work very well with GPT-2 pre-trained models. Maybe I need to play around with them more or maybe they work better with chat-completion and post-trained models. Though, I like the repetition penalty on its own. Most likely, I will stick with multinomial sampling with top-p/top-k as the default recipe.

- [ ] Sampling Strategies II: Beam search implementation
    > Note: For now I will not implement beam search it seems too much of a work for now given it is less popular and practical for decoder-only models.

- [-] Inference Speed Optimization I
    - [ ] **FP16/BF16 Toggle**: Allow reduced precision inference and compare against FP32
    - [ ] **Variable-Length Batching**: Add left-padding with EOS token plus attention masks
    - [ ] **KV Cache Benchmark**: Implement optional cache and measure speedup vs baseline
- [ ] Inference Speed Optimization II: Draft-verify speculative decoding (Try to 
implement)

## Resources

1. For the basic inference implementation:
    - [Karpathy's nanoGPT - Generate Function](https://github.com/karpathy/nanoGPT/blob/master/model.py): Simple and clean implementation of text generation
    - [HuggingFace Transformers - Generation Utils](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py): To understand the sampling and temperature scaling logic
    - [HuggingFace LLM Tutorial](https://huggingface.co/docs/transformers/en/llm_tutorial): Great guide on text generation and decoding strategies

2. For the sampling strategies I:
    - [Chip Huyen's blog post on Generation configs](https://huyenchip.com/2023/03/07/llm-inference.html): Great small blog post top understand Top-K and Top-P sampling strategies
    - [Transformers Library Top-K and Top-P sampling implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py): You can through the `TopPLogitsWrapper` and `TopKLogitsProcessor` to get a feel of how you need to implement the sampling strategies.

3. For the penalty controls:
    - [Transformers Library Repetition Penalty implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py): Try to understand the implementation of repetition penalty from `RepetitionPenaltyLogitsProcessor` class.

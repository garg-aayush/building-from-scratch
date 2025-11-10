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

- [x] Inference Speed Optimization I
    - [x] **FP16/BF16 Toggle**: Allow reduced precision inference and compare against FP32
    - [x] **Implement KV Cache**:
        - [x] Write dynamic kv cache implementation 
        - [x] Pre-allocate memory for the cache (static kv cache)
        - [x] Add a fix for max tokens > block_size (use sliding window) to avoid overflow issue
    - [x] **Variable-Length Batching**: Accept a list of prompts, right-pad with EOS, track per-sample masks, and stop generation per sequence once EOS is reached.

- [x] Inference Speed Optimization II: Draft-verify speculative decoding
    - [x] Implement draft-verify speculative decoding based on the deepmind draft-verify speculative decoding paper
    > Note: I have implemented the speculative decoding implementation in `infer_speculative.py` which do not use the KV cache. The idea is to keep it simple and focus on the speculative decoding logic and speedup.
- [-] Benchmark code
    - [x] Benchmark the speculative decoding speedup vs baseline
    > `benchmark_speculative.py` contains the **vibe-coded** benchmarking code and you can find the results in the `benchmark_results` folder.
    - [ ] Benchmark kv-cache implementation

## Notes

### Using Variable-Length Batching

- **Variable-length batching is only supported when `use_cache=False`**: 
    - When `use_cache=True`, the implementation requires all input sequences to have identical token lengths (after tokenization). This means you can either:
        - Provide a single prompt string
        - Provide the same prompt repeated multiple times (for generating multiple samples)
        - Provide a list of prompts with different lengths but same token length after tokenization
    - If variable-length sequences are detected with `use_cache=True`, the code will turn off the cache and use the normal generation logic.

- **Padding Strategy: Right Padding**
    - Right padding is used because GPT-2 uses absolute positional embeddings, thus the model ties meaning to specific position indices (0, 1, 2, …). Right-padding preserves this alignment, ensuring tokens always start from the same absolute positions the model saw during pre-training.
    - If you left-pad, the real tokens get shifted to higher positions — this breaks the learned positional patterns and can degrade performance. It also complicates generation and attention masking.    


### Speculative Decoding Benchmark Results

I ran a small benchmarking experiment to see whether there is some speedup using speculative decoding for relatively small models like `gpt2-xl` (~1.5B params) and `gpt2-large` (~0.7B params).

**Test Configuration:**
- GPU: rtx 4090, cuda: 12.8, torch: 2.9.0
- Models: draft (**gpt2**) → target: **gpt2-large** / **gpt2-xl**
- tokens/s and acceptance ratio are averaged over 3 runs of 3 prompts with `max_new_tokens=200`.

#### `float16`

<table>
  <thead>
    <tr>
      <th rowspan="2">gamma</th>
      <th colspan="3" style="text-align: center;"> gpt2-large (target), gpt2 (draft)</th>
      <th colspan="3" style="text-align: center;"> gpt2-xl (target), gpt2 (draft)</th>
    </tr>
    <tr>
      <th>baseline (tok/s)</th><th>speculative (tok/s)</th><th>speedup</th>
      <th>baseline (tok/s)</th><th>speculative (tok/s)</th><th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>baseline</td><td>184.29</td><td>—</td><td>1.00x</td><td>136.89</td><td>—</td><td>1.00x</td></tr>
    <tr><td>3</td><td>184.29</td><td>215.59</td><td>1.17x</td><td>136.89</td><td>179.23</td><td>1.31x</td></tr>
    <tr><td>4</td><td>184.29</td><td>201.97</td><td>1.10x</td><td>136.89</td><td>166.47</td><td>1.22x</td></tr>
    <tr><td>5</td><td>184.29</td><td>174.17</td><td>0.95x</td><td>136.89</td><td>153.79</td><td>1.12x</td></tr>
    <tr><td>6</td><td>184.29</td><td>168.94</td><td>0.92x</td><td>136.89</td><td>134.08</td><td>0.98x</td></tr>
    <tr><td>7</td><td>184.29</td><td>164.97</td><td>0.90x</td><td>136.89</td><td>135.56</td><td>0.99x</td></tr>
  </tbody>
</table>

#### `float32`
<table>
  <thead>
    <tr>
      <th rowspan="2">gamma</th>
      <th colspan="3" style="text-align: center;"> gpt2-large (target), gpt2 (draft)</th>
      <th colspan="3" style="text-align: center;"> gpt2-xl (target), gpt2 (draft)</th>
    </tr>
    <tr>
      <th>baseline (tok/s)</th><th>speculative (tok/s)</th><th>speedup</th>
      <th>baseline (tok/s)</th><th>speculative (tok/s)</th><th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>baseline</td><td>106.07</td><td>—</td><td>1.00x</td><td>63.56</td><td>—</td><td>1.00x</td></tr>
    <tr><td>3</td><td>106.07</td><td>154.74</td><td>1.46x</td><td>63.56</td><td>95.80</td><td>1.51x</td></tr>
    <tr><td>4</td><td>106.07</td><td>152.52</td><td>1.44x</td><td>63.56</td><td>98.19</td><td>1.54x</td></tr>
    <tr><td>5</td><td>106.07</td><td>135.28</td><td>1.28x</td><td>63.56</td><td>103.66</td><td>1.63x</td></tr>
    <tr><td>6</td><td>106.07</td><td>132.22</td><td>1.25x</td><td>63.56</td><td>92.78</td><td>1.46x</td></tr>
    <tr><td>7</td><td>106.07</td><td>124.97</td><td>1.18x</td><td>63.56</td><td>90.92</td><td>1.43x</td></tr>
  </tbody>
</table>

#### `bfloat16`
<table>
  <thead>
    <tr>
      <th rowspan="2">gamma</th>
      <th colspan="3" style="text-align: center;"> gpt2-large (target), gpt2 (draft)</th>
      <th colspan="3" style="text-align: center;"> gpt2-xl (target), gpt2 (draft)</th>
    </tr>
    <tr>
      <th>baseline (tok/s)</th><th>speculative (tok/s)</th><th>speedup</th>
      <th>baseline (tok/s)</th><th>speculative (tok/s)</th><th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>baseline</td><td>179.06</td><td>—</td><td>1.00x</td><td>134.00</td><td>—</td><td>1.00x</td></tr>
    <tr><td>3</td><td>179.06</td><td>207.12</td><td>1.16x</td><td>134.00</td><td>163.67</td><td>1.22x</td></tr>
    <tr><td>4</td><td>179.06</td><td>191.35</td><td>1.07x</td><td>134.00</td><td>157.85</td><td>1.18x</td></tr>
    <tr><td>5</td><td>179.06</td><td>170.26</td><td>0.95x</td><td>134.00</td><td>156.11</td><td>1.17x</td></tr>
    <tr><td>6</td><td>179.06</td><td>155.20</td><td>0.87x</td><td>134.00</td><td>130.80</td><td>0.98x</td></tr>
    <tr><td>7</td><td>179.06</td><td>148.57</td><td>0.83x</td><td>134.00</td><td>120.66</td><td>0.90x</td></tr>
  </tbody>
</table>


#### Key Takeaways

1. I have written a basic speculative decoding implementation using the original draft-verify speculative decoding paper setup without much optimization and it shows 1.1–1.6× speedups on small models like `gpt2-large` and `gpt2-xl` showing the potential of speculative decoding. 
2. As mentioned in the original DeepMind paper, the speedup scales with the target-to-draft gap. The larger the target-to-draft gap, the higher the payoff. For example, `gpt2-xl` (~2× `gpt2-large`, >10× `gpt2`) achieves consistent speedups up to 1.6× faster decoding as larger models have higher per-token latency, so skipping even a fraction of target calls yields greater relative savings.
3. Across all settings, gamma = 3/4 yields the best speedups across all precisions and models combinations. This is because at these values the acceptance rate is highest and we see significant speedups.
4. For `float32` precision, we see the highest relative speedups even though their absolute throughput is lowest. This is because the baseline (non-speculative) `float32` inference is much slower, so speculative decoding removes a larger absolute chunk of compute time. 


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

4. For the inference speed optimization I:
    - [Sebastian Raschka's Understanding and Coding the KV Cache in LLMs](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms): Excellent in-depth tutorial on KV cache implementation from scratch with clear explanations and optimizations
    - [Karpathy's nanochat engine.py](https://github.com/karpathy/nanochat/blob/dfc88334b61a3acaf3ec3e61d415d160214f07e9/nanochat/engine.py): A bit difficult to understand the kv-cache part in one go but still a good reference to refer and get intuition on how to go about implementing it.
    - [Umar Jamil's KV Cache Explained](https://www.youtube.com/watch?v=80bIUggRJf4): Video walkthrough of KV cache concept
    - [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4): Video walkthrough of KV cache concept

5. For the speculative decoding (Inference Speed Optimization II):
    - [DeepMind's Draft-Verify Speculative Decoding Paper](https://arxiv.org/abs/2302.01318): The original paper on speculative decoding, it is a great read to understand the concept and the implementation details.
    - [Efficient NLP: Speculative Decoding](https://www.youtube.com/watch?v=S-8yr_RibJ4): Really nice Draft-Target Speculative Decoding Intuitive Explanation
    - [Julian Simon's Optimizing LLM Inference](https://www.youtube.com/watch?v=hMs8VNRy5Ys&t=15s): Introduces Draft-Target, n-gram based, approaches to build draft model including Medusa
    - [Feifei Bear's LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling): Implementation of draft-verify speculative decoding
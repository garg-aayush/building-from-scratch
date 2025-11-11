# LLM Inference from Scratch

This folder contains a from scratch implementation of LLM inference. I built it incrementally starting with greedy decoding and extending to sampling (multinomial, top‑k, top‑p), penalty controls (presence, frequency, repetition), and latency optimizations (fp16/bf16, kv-cache, variable‑length batching, speculative decoding).

## Table of Contents
- [TL;DR](#tldr)
- [How to go through the implementation](#how-to-go-through-the-implementation)
- [Notes](#notes)
  - [Speculative Decoding Benchmark Results](#speculative-decoding-benchmark-results)
  - [KV-Cache Benchmark Results](#kv-cache-benchmark-results)
  - [Using Variable-Length Batching](#using-variable-length-batching)
- [Repository Structure](#repository-structure)
- [Implementation Progress](#implementation-progress)
- [Resources](#resources)

## TL;DR

### Implemented functionalities

**Basic Inference & Sampling:**
- greedy decoding, EOS handling, context window management using sliding window
- temperature scaling, multinomial sampling
- top-k and top-p (nucleus) sampling
- presence, frequency, and repetition penalties controls

**Latency Optimizations:**
- fp16/bf16 optimized inference
- kv-cache (dynamic -> static + overflow fix)
- variable-length batching with right-padding (allows for samples with different lengths)
- draft-verify speculative decoding based on the [DeepMind paper](https://arxiv.org/abs/2302.01318)

#### Benchmark Results 

- config: RTX 4090, cuda 12.8, torch 2.9.0

| Optimization | Best Speedup (float32) | Best Speedup (float16) |
|--------------|------------------------|------------------------|
| kv-cache | **2.76×** (gpt2-large, 800 tokens) | **1.48×** (gpt2-xl, 800 tokens) |
| speculative decoding | **1.63×** (draft: gpt2 -> target: gpt2-xl, gamma=5) | **1.31×** (draft: gpt2 -> target: gpt2-xl, gamma=3) |

- You can see the speedup using kv-cache even for such small models especially for long sequences. However, the speedup is negative or minimal for short sequences where the overhead dominates.
- Speculative decoding works best with gamma=3-4. As expected, the larger target-to-draft gaps leads to higher speedups. Note, you can see the full benchmark results in the [Speculative Decoding Benchmark Results](#speculative-decoding-benchmark-results) section.

## How to go through the implementation

If you want to understand how I built this step-by-step, please explore the commit history. Each commit adds a specific feature, making it easy to learn and go through the code progressively.

### Commit Summary

| Commit ID | Title | Description |
|-----------|-------|-------------|
| `146cc5f` | init | Initial setup and planning |
| `0b911dd` | greedy-decoding | Basic greedy decoding with argmax selection |
| `baeaba3` | eos-handling | Stop generation when EOS token encountered |
| `81265ba` | context-window | Sliding window to keep sequence length ≤ block_size |
| `51dd830` | temperature | Temperature parameter for sampling randomness |
| `a27b92e` | multinomial | Multinomial sampling from full distribution |
| `e22982f` | top-k | Top-k filtering (keep k most likely tokens) |
| `5d7729a` | top-p | Nucleus sampling (cumulative probability threshold) |
| `cb8a30f` | sanity-checks | Parameter validation for temperature, top-k, top-p |
| `0b7ec4e` | presence-penalty | Penalize tokens that already appeared |
| `6290270` | frequency-penalty | Scale penalty by token repetition count |
| `1259e18` | repetition-penalty | Transformers-style repetition penalty |
| `c0a53e0` | fp16-bf16 | Reduced precision inference support |
| `78bfe6b` | kv-cache-dynamic | Dynamic KV cache implementation |
| `551ed2b` | kv-cache-static | Pre-allocated static KV cache |
| `ee24a09` | kv-cache-fix | Sliding window fix for max_tokens > block_size |
| `39a3ce7` | variable-batching | Variable-length batching with right-padding |
| `c721f6a` | speculative-decoding | Draft-verify speculative decoding |
| `c841611` | benchmark-speculative | Benchmark speculative decoding speedup |
| `0e1189a` | benchmark-kv-cache | Benchmark KV cache implementation |

## Notes

### Speculative Decoding Benchmark Results

I ran a small benchmarking experiment to see whether there is some speedup using speculative decoding for relatively small models like `gpt2-xl` (~1.5B params) and `gpt2-large` (~0.7B params). Full results: [`summary_speculative.csv`](benchmark_results/summary_speculative.csv) and [`benchmark_speculative.json`](benchmark_results/benchmark_speculative.json).

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

1. I have written a basic speculative decoding implementation using the original [draft-verify speculative decoding paper](https://arxiv.org/abs/2302.01318) setup without much optimization and even then it shows 1.1–1.6× speedups on small models like `gpt2-large` and `gpt2-xl`. This highlights the potential of speculative decoding. 
2. As mentioned in the original [DeepMind paper](https://arxiv.org/abs/2302.01318), the speedup scales with the target-to-draft gap. The larger the target-to-draft gap, the higher the payoff. For example, `gpt2-xl` (~2× `gpt2-large`, >10× `gpt2`) achieves consistent speedups up to 1.6× faster decoding as larger models have higher per-token latency, so skipping even a fraction of target calls yields greater relative savings.
3. Across all settings, gamma = 3/4 yields the best speedups across all precisions and models combinations. This is because at these values the acceptance rate is highest and we see significant speedups.
4. For `float32` precision, we see the highest relative speedups even though their absolute throughput is lowest. This is because the baseline (non-speculative) `float32` inference is much slower, so speculative decoding removes a larger absolute chunk of compute time. 

### KV Cache Benchmark Results

I also ran a small benchmarking experiment to compare the KV-cache (static sliding window implementation) speedup using [`benchmark_kv_cache.py`](benchmark_kv_cache.py). Full results: [`summary_kv_cache_gpt-large.csv`](benchmark_results/summary_kv_cache_gpt-large.csv) and [`summary_kv_cache_gpt-xl.csv`](benchmark_results/summary_kv_cache_gpt-xl.csv). Note: the speedup is relative to the baseline without KV cache.

#### `bfloat16`

<table>
  <thead>
    <tr>
      <th rowspan="2">max. new tokens</th>
      <th colspan="3" style="text-align: center;">gpt2-large</th>
      <th colspan="3" style="text-align: center;">gpt2-xl</th>
    </tr>
    <tr>
      <th>baseline (tok/s)</th><th>kv cache (tok/s)</th><th>speedup</th>
      <th>baseline (tok/s)</th><th>kv cache (tok/s)</th><th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>200</td><td>165.31</td><td>155.28</td><td>0.94x</td><td>130.16</td><td>121.85</td><td>0.94x</td></tr>
    <tr><td>400</td><td>156.98</td><td>153.71</td><td>0.98x</td><td>110.17</td><td>117.19</td><td>1.06x</td></tr>
    <tr><td>800</td><td>130.82</td><td>161.12</td><td>1.23x</td><td>80.72</td><td>111.27</td><td>1.38x</td></tr>
  </tbody>
</table>

#### `float16`
<table>
  <thead>
    <tr>
      <th rowspan="2">max. new tokens</th>
      <th colspan="3" style="text-align: center;">gpt2-large</th>
      <th colspan="3" style="text-align: center;">gpt2-xl</th>
    </tr>
    <tr>
      <th>baseline (tok/s)</th><th>kv cache (tok/s)</th><th>speedup</th>
      <th>baseline (tok/s)</th><th>kv cache (tok/s)</th><th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>200</td><td>175.85</td><td>160.51</td><td>0.91x</td><td>122.44</td><td>112.20</td><td>0.92x</td></tr>
    <tr><td>400</td><td>159.03</td><td>149.52</td><td>0.94x</td><td>106.06</td><td>112.14</td><td>1.06x</td></tr>
    <tr><td>800</td><td>128.57</td><td>169.30</td><td>1.32x</td><td>86.07</td><td>127.12</td><td>1.48x</td></tr>
  </tbody>
</table>

#### `float32`
<table>
  <thead>
    <tr>
      <th rowspan="2">max. new tokens</th>
      <th colspan="3" style="text-align: center;">gpt2-large</th>
      <th colspan="3" style="text-align: center;">gpt2-xl</th>
    </tr>
    <tr>
      <th>baseline (tok/s)</th><th>kv cache (tok/s)</th><th>speedup</th>
      <th>baseline (tok/s)</th><th>kv cache (tok/s)</th><th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>200</td><td>103.48</td><td>174.02</td><td>1.68x</td><td>62.51</td><td>104.41</td><td>1.67x</td></tr>
    <tr><td>400</td><td>76.61</td><td>163.48</td><td>2.13x</td><td>48.64</td><td>99.89</td><td>2.05x</td></tr>
    <tr><td>800</td><td>53.82</td><td>148.34</td><td>2.76x</td><td>38.09</td><td>93.03</td><td>2.44x</td></tr>
  </tbody>
</table>

#### Key Takeaways

1. You can see that for short generations, speedups are limited or negative. This is due to the device transfer and communication overheads dominating the runtime. As the generated length grows, reuse increases and you start seeing the benefits of the kv-cache.
2. Like in speculative decoding, the speedups are strongest in `float32` where baseline compute is slowest, so avoiding recomputation yields the largest relative gains. In `bfloat16/float16` case, optimized GPU kernels and lower per-step latency mean kv-cache overheads can dominate until sequences are long enough.

### Using Variable-Length Batching

- **Variable-length batching is only supported when `use_cache=False`**: 
    - When `use_cache=True`, the implementation requires all input sequences to have identical token lengths (after tokenization). This means you can either:
        - Provide a single prompt string
        - Provide the same prompt repeated multiple times (for generating multiple samples)
        - Provide a list of prompts with different lengths but same token length after tokenization
    - If variable-length sequences are detected with `use_cache=True`, the code will turn off the cache and use the normal generation logic.

- **Padding Strategy: Right Padding**
    - I have used right padding because GPT-2 uses absolute positional embeddings, thus the model ties meaning to specific position indices (0, 1, 2 etc). Right-padding preserves this alignment, ensuring tokens always start from the same absolute positions the model saw during pre-training.
    - If you left-pad, the real tokens get shifted to higher positions. This breaks the learned positional patterns and can degrade performance. It also complicates generation and attention masking.

## Repository Structure

```
llm-inference/
├── model.py                      # GPT-2 model implementation (from gpt-2 code written before)
├── infer.py                      # Main inference implementation with all features
├── infer_speculative.py          # Speculative decoding implementation
├── benchmark_kv_cache.py         # vibe-coded KV cache benchmarking script 
├── benchmark_speculative.py      # vibe-coded speculative decoding benchmarking script
│
├── benchmark_results/            # Benchmark results (CSV files)
│
└── Readme.md                     # This file
```


## Implementation Progress

My starting point for the implementation is [`model.py`](model.py) from the GPT-2 work (commit: `a100995` on branch gpt2).

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
    > Note: I have implemented all three penalties in [`infer.py`](infer.py). However, tbh I don't think they work very well with GPT-2 pre-trained models. Maybe I need to play around with them more or maybe they work better with chat-completion and post-trained models. Though, I like the repetition penalty on its own. Most likely, I will stick with multinomial sampling with top-p/top-k as the default recipe.

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
    - [x] Implement draft-verify speculative decoding based on the [DeepMind draft-verify speculative decoding paper](https://arxiv.org/abs/2302.01318)
    > Note: I have implemented the speculative decoding implementation in [`infer_speculative.py`](infer_speculative.py) which do not use the KV cache. The idea is to keep it simple and focus on the speculative decoding logic and speedup.
- [x] Benchmark code
    - [x] Benchmark the speculative decoding speedup vs baseline
    > [`benchmark_speculative.py`](benchmark_speculative.py) contains the **vibe-coded** benchmarking code and you can find the results in the [`benchmark_results`](benchmark_results/) folder.
    - [x] Benchmark kv-cache implementation
    > [`benchmark_kv_cache.py`](benchmark_kv_cache.py) contains the benchmarking code and results are in the [`benchmark_results`](benchmark_results/) folder.

## Resources

1. For the basic inference implementation:
    - [Karpathy's nanoGPT - Generate Function](https://github.com/karpathy/nanoGPT/blob/master/model.py): Simple and clean implementation of text generation
    - [HuggingFace Transformers - Generation Utils](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py): To understand the sampling and temperature scaling logic
    - [HuggingFace LLM Tutorial](https://huggingface.co/docs/transformers/en/llm_tutorial): Great guide on text generation and decoding strategies

2. For the sampling strategies I:
    - [Chip Huyen's blog post on Generation configs](https://huyenchip.com/2023/03/07/llm-inference.html): Great small blog post to understand top-k and top-p sampling strategies
    - [Transformers Library Top-K and Top-P sampling implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py): You can go through the `TopPLogitsWrapper` and `TopKLogitsProcessor` to get a feel of how you need to implement the sampling strategies.

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

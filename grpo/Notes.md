# Running Notes for GRPO from scratch

Following the same approach as I did for the SFT (supervised fine-tuning) and GPT-2 pre-training code from scratch, I have written the GRPO code from scratch. Again, I followed the 
Stanford CS336 [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_alignment.pdf) as a reference point and trained Qwen2.5-Math-1.5B with verifiable math rewards. This time around I had 3 main motivations:

1. As usual, write the GRPO code from scratch for the sake of understanding.
2. Train Qwen2.5-Math-1.5B with verifiable math rewards and get a feel of what kind of accuracy we can push with pure RL (no supervised fine-tuning)
3. **Most importantly**, run a lot of ablation studies to understand and build intuition on what matters in GRPO training, the different design choices we can make and how to interpret the different metrics. To be honest, now I look back I think this is the most important part of this long exercise. 

A quick recap on what GRPO is:
- [GRPO](https://huggingface.co/blog/garg-aayush/derive-grpo-loss) (Group Relative Policy Optimization) is a RL algorithm that eliminates the need for a separate critic/value model by using group-relative advantages for each prompt. It generates multiple candidate outputs, scores them and normalizes rewards within the group to get advantages. GRPO instead of imitating expert reasoning traces (SFT), it lets the model discover its own strategies by generating multiple candidate solutions per problem, scoring them and reinforcing the better ones.

> You can read more about it in the [GRPO derivation blog post](https://huggingface.co/blog/garg-aayush/derive-grpo-loss) I wrote a few weeks back and in this [blog post](https://substack.com/home/post/p-177823868)


## Building the Training Loop

I followed the same approach as I did for SFT following the assignment where I write and test the helper functions first and make sure each piece works in isolation. Finally, wiring all the helper functions together into the full training loop. 

The GRPO algorithm has two nested loops:
- **Outer loop**: sample a batch of prompts, generate G `rollouts` per prompt, compute rewards` normalize advantages within each group
- **Inner loop**: policy gradient updates over the rollout batch (obviously using the gradient accumulation trick to fit on the GPU)

For the GRPO training, you only need the MATH dataset without any reasoning traces. I sourced the data from the [CS336 MATH dataset](https://github.com/kkaitlyn111/cs336-a5-RL/tree/main/MATH) repo.

> I have made sure to use the functions, data pipelines etc from SFT code where ever possible.

### Helper functions (`utils/grpo.py`)

You can find the main GRPO driving functions in the [`utils/grpo.py`](utils/grpo.py) file. The important ones are:
- `compute_group_normalized_rewards`: takes per-rollout rewards and normalizes advantages within each group (subtract group mean, optionally divide by group std)
- Four loss functions:
  - `no_baseline` — plain REINFORCE: just multiply log probs by rewards
  - `reinforce_with_baseline` — subtract the group mean reward before multiplying
  - `grpo_clip` — clipped PPO-style ratio objective
  - `grpo_no_clip`
- `grpo_microbatch_train_step` — single microbatch forward/backward pass with loss computation

### Training script (`train_grpo.py`)

The [`train_grpo.py`](train_grpo.py) file contains the main training loop where the outer and inner loops are wired together:
1. Sample prompts from training data
2. Generate G rollouts per prompt via vLLM
3. Compute rewards for each rollout
4. Normalize advantages within each group
5. Compute old log probs (for ratio-based losses)
6. Run inner training loop: iterate over rollouts in microbatches, accumulate gradients and optimize.

### Important Notes:

- I used the same vLLM colocate setup and workarounds as I did for SFT (see the [SFT Notes](../sft/Notes.md) for more details). This allowed me to run the training loop and intermediate evaluations and rollout generations on a single GPU.
- I have also added the intermediate evaluation on the validation set, model checkpointing at configurable intervals, wandb logging, eval and timing metrics for better observability.
- All training configs are managed via [OmegaConf](https://omegaconf.readthedocs.io/) structured configs and yaml files. This is especially useful for ablation studies where each experiment config is a minimal diff from the defaults, making it easy to see exactly what changed and importantly to reproduce any run.

> Given the compute constraints (I am paying out of my own pocket for the compute), all intermediate evaluations during training are done on a subset of 1024 examples from the validation set, not the full ~5K. This keeps the evaluation fast enough to run every few GRPO and at the end.


## GPU Memory Optimization (Fitting on 24 GB)

As I proceed ahead with writing the training scripts with the default parameters suggested in the assignment, one of the issues that I ran into was the memory constraints (OOMs error!). I have ben writing the scripts and testing them on my personal RTX 4090 with 24 GB of VRAM. Thus, I actually had to do some memory optimizations to fit the training loop. 

- **Peak memory tracking** (`track_peak_memory`, commit `f878a77`): This is not an optimization in itself but an essential step to keep track of the peak memory usage. I made sure to log the peak memory at important junctures in the training loop.
- **Gradient checkpointing** (`use_gradient_checkpointing`, commit `c2bce8d`): the simplest one is to enable gradient checkpointing which recomputes activations during backward pass instead of storing them. You can trade upto ~30% memory saving savings for speed.
- **vLLM sleep mode** (`use_vllm_sleep_mode`, commit `4ddca51`): this is quite a nice trick to offload vLLM KV cache and weights to CPU during the training phase (when vLLM is not generating). This frees GPU memory for the backward pass and prevents the vLLM cache from competing with training activations.
- **8-bit AdamW** (`use_bnb_adamw8bit`, commit `eb52b83`): I added an option to use bitsandbytes `AdamW8bit` optimizer instead of the default `AdamW` optimizer. This reduces the optimizer state memory by almost half.


After all the above optimizations tricks, I was able to successfully train with `rollout_batch_size=256`, `group_size=8`, `gradient_accumulation_steps=256` (microbatch size = 1) on 24 GB. You can find the reference config in the [`configs/test.yaml`](configs/test.yaml). Moreover, the default config in the [`configs/defaults.py`](configs/defaults.py) file is optimized for the RTX 4090.

> Note: You should be able to run most of the ablation studies on your local RTX 4090 with the default configs optimization flags.


## Scaling to Modal (H100)

The local training script runs fine on a 24 GB GPU (e.g. RTX 4090) but there are two practical limitations that made me want to scale to Modal:

1. **Speed**: even with all memory optimizations, a single experiment on the 4090 takes a few hours. Some of the ablation studies and hyperparameter searches become impractical.
2. **Parallelism**: I wanted to run a lot of experiments like LR sweeps, baseline comparisons, off-policy sweeps, etc. On a single local GPU that means running them one at a time which would take weeks. Thus, I needed a way to fire off multiple experiments in parallel and compare them side-by-side in wandb. At the same time, I am not spending my full time on this and I work on it whenever I get time. Thus, I did not want to deal with spinning up and tearing down GPU instances each time.

[Modal](https://modal.com/) is a great AI infra platform with lots of available GPU (A100, H100, H200) instances where you define your workload in Python (no Docker/Kubernetes). On Modal, you get pay-per-second billing so **you only pay for actual compute time**. The containers spin up in seconds, you can fire off multiple H100 runs simultaneously and everything scales back to zero when you are done. You define everything in Python with decorators and Modal handles container images, secrets, and persistent volumes. It is a great platform to spin up/down a GPU for training and evaluation in seconds. 

### One-time setup (run from inside `grpo/`)

```bash
# Create Modal secrets (once)
modal secret create wandb-secret WANDB_API_KEY=<your-key>
modal secret create huggingface-secret HF_TOKEN=<your-token>   # optional

# Download model weights into the grpo-data volume
modal run utils/setup_modal.py::setup_model

# Upload training / validation data into the grpo-data volume
modal run utils/setup_modal.py::upload_data --local-data-dir /path/to/DATA/GRPO
```

### Launching a training run

```bash
# Basic launch 
modal run train_on_modal.py --config configs/test_modal.yaml

# Always set --output-dir to a unique path so runs dont overwrite each other
modal run train_on_modal.py --config configs/test_modal.yaml --output-dir /results/test_modal

# You can submit the job and free the terminal immediately using the --detach and --spawn flags.
modal run --detach train_on_modal.py --config configs/test_modal.yaml --output-dir /results/test_modal --spawn
```

`--output-dir` must point inside the `/results/` volume mount so checkpoints and logs are persisted after the container exits.

### H100 config optimization

One more thing I did was to optimize the config for the H100. I disabled the memory tricks that exist only to fit on a 24 GB card and used larger microbatches and fewer old log probs passes. You can find the reference config in the [`configs/test_h100_modal.yaml`](configs/test_h100_modal.yaml) file.

| Flag | Default | H100 | Why |
|---|---|---|---|
| `use_gradient_checkpointing` | `true` | `false` | No need to recompute activations which results in faster backward pass |
| `use_bnb_adamw8bit` | `true` | `false` | Fused float32 AdamW is faster on H100 |
| `gradient_accumulation_steps` | `256` (microbatch=1) | `64` (microbatch=4) | Larger microbatches result in better tensor core utilization |
| `old_log_probs_train_size` | `2` | `4` | Fewer and larger passes for old log probs |
| `use_vllm_sleep_mode` | `true` | `true` | I still keep this flag on for the H100 since it is still useful to free KV cache during training when vLLM is not generating.|

**Timing comparison (20 GRPO steps, `reinforce_with_baseline`):**

| Hardware | Config | Time |
|---|---|---|
| RTX 4090 (24 GB) | RTX 4090 defaults | ~28 min |
| H100 (80 GB) | RTX 4090 defaults (unchanged) | ~18 min |
| H100 (80 GB) | H100-optimized (`test_h100_modal.yaml`) | ~10 min |

> **Cost note:** Running all the ablation studies discussed in the ablation studies section including failed experiments and runs I terminated early, costed approximately **$140** on Modal. I think that is well worth it for the understanding I gained. 


## Configuration, Metrics and Experiment Tracking

### Default configuration

All defaults live in [`configs/defaults.py`](configs/defaults.py) as OmegaConf structured dataclasses. Yaml config files only need to specify overrides.

**Key GRPO parameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_grpo_steps` | 10 | Number of outer GRPO steps |
| `rollout_batch_size` | 256 | Total rollouts per GRPO step |
| `group_size` | 8 | Rollouts per prompt (so `rollout_batch_size / group_size` = 32 prompts/step) |
| `epochs_per_rollout_batch` | 1 | On-policy if 1, off-policy if > 1 |
| `train_batch_size` | 256 | Batch size for the inner training loop |
| `gradient_accumulation_steps` | 256 | Microbatch size = `train_batch_size / gradient_accumulation_steps` |
| `loss_type` | `grpo_clip` | Loss variant (see below) |
| `normalize_mode` | `mean` | How will the per-token losses be aggregated: `mean`, `constant` or `microbatch` |
| `use_std_normalization` | `True` | it divides the advantages by group std (Shao et al.) |

**Loss types:**
- `no_baseline`: plain REINFORCE; raw reward as advantage
- `reinforce_with_baseline`: subtract group mean reward before computing gradients
- `grpo_clip`: clipped PPO-style ratio
- `grpo_no_clip`: unclipped ratio

### Metrics tracked

I track 8 metrics across training and evaluation. The most important ones to watch are `eval/reward` (is the model actually getting better?), `train/grad_norm` (early warning for instability, look for spikes here which predict reward collapse), and `train/mean_response_length` (tells us how average rollouts responses length, a stable training should see this slowly increasing over time).

| Metric | What it measures |
|---|---|
| `eval/reward` | Validation accuracy (fraction of correct math answers) |
| `eval/format_reward` | Format accuracy (correct output format structure) |
| `train/loss` | Policy gradient loss |
| `train/grad_norm` | Gradient norm after clipping |
| `train/mean_response_length` | Average rollouts response length in tokens |
| `train/entropy` | Entropy of next-token predictions |
| `train/mean_ratio` | `pi_theta / pi_theta_old` should be ~1.0 on-policy |
| `train/clip_fraction` | Fraction of tokens where clipping changed the objective |

**Notes on `train/mean_ratio` and `train/clip_fraction`:**
- `train/mean_ratio` measures policy drift from the rollout distribution. It should stay near 1.0 with `epochs_per_rollout_batch=1` but will drift as inner epochs increase.
- `train/clip_fraction` checks if the clipping actually changed the objective. It does this by checking if `scores != clipped_scores`, not just `ratio outside [1-eps, 1+eps]`.
- For `grpo_no_clip` and `reinforce_*` loss types, `train/clip_fraction` is always 0.0 which is expected.

### Timing metrics

I also logged the timing metrics for each GRPO step. This helps me understand where the time is spent and if there are any bottlenecks.

| Metric | What it measures |
|---|---|
| `rollout_dt` | Time for vLLM to generate all rollouts |
| `train_dt` | Accumulated time across all inner train steps (all epochs x all microbatches). Each inner step is forward + loss + grad accum + clip_grad_norm + optimizer.step() |
| `eval_dt` | It is the time for the full `evaluate_vllm()` call. Zero when no eval runs at this step |
| `step_dt` | Total wall time for the GRPO step (covers rollout + tokenization + old log probs + training). It does **not** include eval or weight-loading into vLLM |

Note: `rollout_dt + train_dt <= step_dt` since `step_dt` also includes tokenization, old log prob computation, and vLLM sleep/wake overhead.

### W&B

All experiments at [wandb.ai/garg-aayush/grpo](https://wandb.ai/garg-aayush/grpo).


## Ablation Studies

I ran a series of ablation studies as per the assignment to understand what matters in GRPO training. Each ablation isolates one design choice while keeping everything else fixed.

You will notice the experiments vary in length: some run for 200 GRPO steps, some for 100, some for only 50. This is intentional. For example, when doing a broad LR or off-policy sweep, I run for a small number of steps first to see what works and what does not. Moreover, bad configs reveal themselves early thus it is better to terminate the run midway when I could see things going sideways or when the result was already clear. **This kept the total cost manageable while still getting the information I needed.**

### Learning Rate Sweep

The learning rate is the most critical hyperparameter to get right first. It determines whether the policy updates are large enough to learn but not so large to cause the policy to collapse. Moreover, unlike supervised learning where a bad `lr` just causes loss divergence, in GRPO a high `lr` can cause the policy to collapse onto degenerate outputs before learning anything useful.

In order, to find the right `lr`, I ran a log spaced search from `1e-6` to `1e-4` for 100 steps each on H100. You can find the reference config in the [`configs/lr_sweep/`](configs/lr_sweep/) directory.

![eval/reward and token entropy across LRs](results/lr_sweep/lr_sweep.png)

- `1e-6` and `3e-6` barely move the eval reward accuracy (`eval/reward`) metric. The gradient signal is too small to update policy meaningfully.
- `1e-4` shows policy collapse with mean response length (`train/mean_response_length`) spikes and token entropy drops to near zero.
- For `1e-5` to `3e-5`, the reward rises steadily, response length stabilizes and token entropy drops steadily.

Thus, I decided to use `3e-5` as the learning rate for the future runs given it gives the most stable training and highest reward accuracy.


### Baseline Ablation

The vanilla REINFORCE gradient has notoriously high variance. A common technique is to subtract a baseline (the group mean reward) from the advantage which reduces variance without introducing bias. Here we tested whether that variance reduction actually matters in practice and looked at whether subtracting the group mean make a measurable difference or is the raw reward signal sufficient?

I ran three runs as shown in the plots. You can find the reference configs in the [`configs/baselines/`](configs/baselines/) directory.

![Baseline ablation: eval reward accuracy, gradient norm, mean response length](results/baselines/baseline_ablation.png)

- `reinforce_with_baseline` evaluation reward accuracy steadily climbs to ~0.61 with stable gradient norm and consistent mean response length around 300-350 tokens.
- However, both `no_baseline` runs peak early then decline. Their gradient norm is way high and seems to keep increasing and both suffer rapid response length collapse after some steps.

This clearly shows that subtracting the group mean reward from the advantage reduces variance and prevents response length collapse. Thus, `reinforce_with_baseline` is a better choice.


### Length Normalization

When aggregating per-token losses over the sequence dimension, the choice of normalization affects how much gradient signal each individual token receives. As noted in the assignment, it is not necessary or even correct to always average losses by sequence length. I tested the three modes: 

- `mean`: divide by number of response tokens per sequence, short correct answers get disproportionately large per-token gradients
- `constant`: divide by a fixed constant like `max_gen_len`=1024, every token gets the same gradient magnitude regardless of sequence length, used in DeepSeek. 
- `microbatch`: normalize by the longest response in the current microbatch which is a middle ground between `mean` and `constant`.

You can find the reference configs in the [`configs/length_normalization/`](configs/length_normalization/) directory.

![Length normalization ablation: eval reward, gradient norm, mean response length](results/length_normalization/length_normalization.png)

All three modes converge to similar final reward accuracy and mean response length. The main difference is in gradient norm. `constant` produces consistently lower norms in comparison to `microbatch` and `mean`. This is expected since `constant` divides everything by 1024, which is 2-2.5x larger than typical response length. 

Overall, length normalization mode has minimal impact on final reward for math reasoning with binary reward. The primary observable difference is in gradient scale and not learning dynamics. I kept `mean` as the default.


### Standard Deviation Normalization

The standard GRPO advantage computation divides by the group standard deviation: `advantage_i = (reward_i - mean(group)) / (std(group) + eps)`. But Liu et al. (2025) ([Dr. GRPO](https://arxiv.org/abs/2503.20783)) argued that this can introduce unwanted biases where too easy or too hard questions rollouts with low variance produce near-zero std deviation and inflating their advantages disproportionately. They proposed removing the division entirely. This ablation tests whether removing that division actually helps.

You can find the reference config in the [`configs/std_dev/`](configs/std_dev/) directory.

![std dev normalization ablation](results/std_dev/std_dev_normalization.png)

- With std normalization reaches higher final reward accuracy (~0.72) while without plateaus at ~0.65. The gap is consistent throughout training.
- Removing std normalization actually improves gradient stability where we have lower gradient norms with less variance. This sort of confirms the observation that dividing by group std amplifies gradients for low-variance groups. However, the improved stability doesn't translate to better performance here.

Give, the std normalization gives ~0.07 higher reward accuracy despite less stable gradients. I decided to keep it. The reward gap is substantial enough to justify the slightly noisier gradients.

### Off-Policy Sweep

On-policy is theoretically clean but highly inefficient, we do a lot of expensive inference to generate rollouts and only to take a single gradient step. The idea here is if we take multiple gradient steps per rollout batch, how far can we push this before the policy drifts too far from the rollout distribution and training becomes unstable? And also, thus off-policy training results in better performance at the cost of extra compute.

You can find the reference configs in the [`configs/off_policy_sweep/`](configs/off_policy_sweep/) directory.

#### Broad sweep (50 steps)

First, I ran a broad sweep over 6 configs varying `epochs_per_rollout_batch` and `train_batch_size` ranging from on-policy (1 optimizer step per GRPO step) to aggressive off-policy (16 optimizer steps per GRPO step). All runs use `grpo_clip` loss , LR=3e-5.

![Broad off-policy sweep: eval reward, gradient norm, mean response length](results/off_policy_sweep/off_policy_sweep.png)

- Most configs converge to ~0.55-0.65 . The clear outlier is `e4_tb64_ga16` (16 opt steps/GRPO). It collapses mid-way with gradient norm spikes and response length collapse to ~100 tokens. This is a classic failure mode where the policy drifts too far from rollout distribution and the model learns to produce minimal outputs.
- Mild off-policy (2 opt steps/GRPO) works as well as on-policy.

#### Full sweep (200 steps)

I then selected the three most promising configs (on-policy, and two mild off-policy) for full 200-step training.

![Full off-policy sweep: eval reward, entropy, gradient norm, mean response length](results/off_policy_full_sweep/off_policy_full_sweep.png)

- On-policy (`e1_tb256_ga64`) is consistently the best. It converges fastest and maintains highest reward accuracy (~0.65-0.75). The two mild off-policy configs track slightly behind and converge toward ~0.65-0.70 by the end.
- `e2_tb256_ga64` (2 epochs) shows higher gradient norm variance with spikes  but doesn't destabilize.
- An interesting side note: on-policy `grpo_clip` is numerically equivalent to `grpo_no_clip` with 1 opt step, the policy ratio is ~1.0 so clipping never fires.

On-policy training is the clear winner. Reusing rollouts doesn't help and the extra compute per GRPO step is not justified by the performance gain. I went with on-policy (`e1_tb256_ga64`).


### Prompt Template Ablation

Here, I compared the `r1_zero` prompt (structured `<think>...</think>` and `<answer>...</answer>` blocks) against question-only (just `{question}`, each with a matching reward function.

![Prompt template ablation: eval reward, entropy, mean response length](results/prompt_ablation/prompt_ablation.png)

- Question-only starts with much higher accuracy because Qwen2.5-Math-1.5B seems to be pre-trained on math data with `\boxed{}` formatting. It already solves nearly half the problems out of the box. In comparison, r1-zero starts near zero (unfamiliar format) but catches up quickly with the help of the structured prompt. Finally, r1-zero consistently performs better than question-only by the end.
- If you look at the entropy plot, it is the most revealing metric here. r1-zero prompt settles at much lower entropy in comparison to question-only. This is expected since r1-zero prompt is more structured and constrains the output space as a result we also get a much sharper final policy.

The structured prompt provides a modest but consistent accuracy advantage as it provides a dedicated reasoning scratchpad for the model to reason before committing to an answer. Without this, reasoning is interleaved with the answer in less predictable ways.


### SFT Checkpoint Initialization

This is not the part of the assignment but I thought it would be interesting to see how starting from an SFT checkpoint affects the performance as it is a natural question to ask. As we already have an SFT model that gets ~53% accuracy, can GRPO push it even higher? Or does starting from a pre-narrowed distribution actually hurt RL ability to explore and find better strategies?

I ran five runs: base model (no SFT), three SFT checkpoints (early/mid/final) and final with lower LR. You can find the reference configs in the [`configs/sft_grpo/`](configs/sft_grpo/) directory.

![SFT -> GRPO sweep: eval reward, format reward, entropy](results/sft_grpo/sft_grpo.png)

- Base model (no SFT) starts still performs better than the SFT runs. As we use more and more SFT checkpoints, the GRPO ceiling plateaus at lower and lower accuracy. Similarly, we see higher entropy in the SFT runs in comparison to the base model.
- This indicates that the SFT checkpoints are not helping the GRPO training and are actually hurting it. This is not surprising since the SFT checkpoints are not distilled from a larger model and are not self-generated traces. Thus, the SFT checkpoints are not able to provide a good starting point for the GRPO training. 
- The base model starting point is able to explore a wider strategy space and converge on better strategies in comparison to the SFT checkpoints.
- Lowering the learning rate on the final SFT checkpoint helps only marginally. The bottleneck is distribution narrowing from SFT not the gradient step size.


## Summary & Key Takeaways

**Best configuration:** 
- on-policy
- lr=`3e-5`
- loss_type=`grpo_clip`
- use_std_normalization=`True`
- r1_zero prompt
- base model (no SFT)


**Best performance:** ~0.75 on MATH validation (up from ~3% base model accuracy). This is the run `off_policy_full_e1_tb256_ga64` with the above config.

![Best GRPO run: eval reward accuracy and mean response length](results/best_run/best_run.png)

**Key lessons:**
- **eval/reward** is the most important metric to watch. It is the metric that matters the most. Especially in the case of reinforcement learning with verifiable rewards.
- However, **gradient norm** and **mean_response_length** are the two other important metrics to watch. They are the early warning signals for instability and reward collapse.
- **Binary math reward is robust to some design choices** (length normalization) **but sensitive to others** (baseline, learning rate).
- **On-policy training seems to work better** for this task.
- **The structured R1-Zero prompt helps improve the accuracy** over question-only by creating a dedicated reasoning scratchpad that constrains the output space.

Ideally, I should now try to push the accuracy further by training for longer, using curriculum strategies or modifying the GRPO loss itself. But that's for another time! I think it is enough learning and compute expenditure for now.

## Commit History

### Phase 1: Core Implementation

| Commit | Title | Description |
|--------|-------|-------------|
| `58ecd8c` | GRPO: initialize grpo folder | Initial project setup |
| `877ea05` | GRPO: add compute_group_normalized_rewards fn | Group-level advantage normalization |
| `aa63db1` | GRPO: add naive policy gradient loss | `no_baseline` loss function |
| `e69867e` | GRPO: add grpo clip loss | Clipped and unclipped ratio losses |
| `da8b36e` | GRPO: add policy gradient wrapper fn | Unified loss dispatch |
| `9f76d6b` | GRPO: add masked_mean and masked_normalize helper fns | Per-token loss aggregation helpers |
| `d0d6e00` | GRPO: add micro_batch train step | Single microbatch forward/backward |
| `3f939ac` | GRPO: debug grpo util fns | Fix issues found during testing |
| `bf2cfdf` | GRPO: update microbatch step function docstring | Documentation |
| `34e2d28` | GRPO: add paths config | OmegaConf structured config |
| `3b5abe8` | GRPO: update train script, init vllm, model and optimizer | Model and vLLM initialization |
| `6fe17fb` | GRPO: update train script, load datasets and prompt template | Data loading |
| `e34040d` | GRPO: make defaults dict simpler | Config cleanup |
| `c6a3cdf` | GRPO: write outer loop except old log prob calculations | Main training loop skeleton |
| `e231437` | GRPO: update pretty print and tokenize train data | Data preprocessing |
| `6a5d388` | GRPO: add code to calculate old log probs | Old policy log probs for ratio losses |
| `c96b970` | GRPO: Write first working version of GRPO train loop | End-to-end training (no eval) |
| `f878a77` | GRPO: track peak memory usage | `[MEM]` logging infrastructure |
| `c2bce8d` | GRPO: add gradient checkpointing flag | Memory optimization #1 |
| `4ddca51` | GRPO: add vllm sleep mode flag | Memory optimization #2 |
| `eb52b83` | GRPO: add option to use adam8bit | Memory optimization #3 |
| `e1dcc23` | GRPO: update config and track mean_response_length | Response length metric |
| `64867a5` | GRPO: add intermediate evaluations | Validation during training |
| `deaa5b8` | GRPO: save model at intermediate steps | Checkpointing |
| `312de9a` | GRPO: log rollouts | Save rollouts to disk for inspection |
| `347608b` | GRPO: track clip fraction and mean ratios | Policy drift metrics |
| `6499c97` | Add wandb logging | W&B experiment tracking |
| `106bd20` | GRPO: add wandb tags and test config | Config for local testing |
| `00db81c` | GRPO: add per-step timing breakdown | `[DT]` timing metrics |

### Phase 2: Modal Setup

| Commit | Title | Description |
|--------|-------|-------------|
| `28390e3` | GRPO: add scripts for running training on modal | `setup_modal.py` + `train_on_modal.py` |
| `d550bcc` | GRPO: fix modal training script bug | Bug fix |
| `026f4cc` | GRPO: add H100-optimized Modal config | `test_h100_modal.yaml` + vllm fix |

### Phase 3: Ablation Experiments

| Commit | Title | Description |
|--------|-------|-------------|
| `c179dbe` | GRPO: run lr-sweep experiments | LR sweep configs + results |
| `33485b5` | GRPO: compare baselines experiments | Baseline ablation configs + results |
| `a2cb59a` | GRPO: run length normalization ablation studies | Length norm configs + results |
| `db6b623` | GRPO: group standard dev. normalization ablation | Std dev normalization experiment |
| `08e1091` | GRPO: off-policy ablation study | Off-policy sweep configs + results |
| `5c072b7` | GRPO: prompt template ablation study | R1-Zero vs question-only |
| `21a962b` | GRPO: SFT checkpoint initialization ablation study | SFT->GRPO experiments |

### Phase 4: Plotting and Uploading
| `3330d85` | GRPO: add plots for baseline, length normalization and LR sweep | Plot scripts + figures |
| `3097865` | GRPO: add plots for std-dev, off-policy and prompt ablation | Plot scripts + figures |
| `7053afa` | GRPO: add SFT checkpoint initialization experiments plots | Plot scripts + figures |
| `717434d` | GRPO: add best run plot and script, update the lr sweep scripts and plots | `plot_best_run.py`, `summarize_runs.py`, updated lr sweep plot + CSVs |
| `a84bc5a` | move modal setup and checkpoint uploads script to grpo/utils | Moved `setup_modal.py` and `upload_to_hf.py` into `grpo/utils/` |

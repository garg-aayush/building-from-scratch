## cmds
grep '^\[MEM\]' log.txt
uv run train_grpo.py 2>&1 | tee logs/log_peak_mem.txt


## Changes for GPU vram optimization

### Start tracking the Peak VRAM
commit: `f878a7783fa2e465c3415732b2378f39f9d2b8f6`
- flag: `track_peak_memory` in `configs/defaults.py`

config:
```python
# GRPO parameters
n_grpo_steps: int = 10                             # number of GRPO steps
advantage_eps: float = 1e-6                         # epsilon for advantage normalization
rollout_batch_size: int = 16                       # number of rollouts per batch
group_size: int = 4                                 # size of each group
epochs_per_rollout_batch: int = 1                   # On-policy (off-policy if > 1)
train_batch_size: int = 16                         # On-policy, batch size for training the policy
gradient_accumulation_steps: int = 16              # microbatch size is 2
```

log:
```
[MEM] after init (model + vLLM + optimizer): current=6.19GB  peak_since_reset=6.19GB
[MEM] [GRPO step 000/009] after rollout generation: current=6.19GB  peak_since_reset=6.22GB
[MEM] [GRPO step 000/009] before training inner loop: current=6.19GB  peak_since_reset=9.49GB
[MEM] [GRPO step 000/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=15.22GB
[MEM] [GRPO step 001/009] after rollout generation: current=11.95GB  peak_since_reset=11.99GB
[MEM] [GRPO step 001/009] before training inner loop: current=11.95GB  peak_since_reset=15.44GB
[MEM] [GRPO step 001/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=21.38GB
[MEM] [GRPO step 002/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 002/009] before training inner loop: current=11.95GB  peak_since_reset=15.17GB
[MEM] [GRPO step 002/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=20.86GB
[MEM] [GRPO step 003/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 003/009] before training inner loop: current=11.95GB  peak_since_reset=15.21GB
[MEM] [GRPO step 003/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=20.91GB
[MEM] [GRPO step 004/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 004/009] before training inner loop: current=11.95GB  peak_since_reset=15.27GB
[MEM] [GRPO step 004/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=21.01GB
[MEM] [GRPO step 005/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 005/009] before training inner loop: current=11.95GB  peak_since_reset=15.47GB
[MEM] [GRPO step 005/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=21.43GB
[MEM] [GRPO step 006/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 006/009] before training inner loop: current=11.95GB  peak_since_reset=13.56GB
[MEM] [GRPO step 006/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.85GB
[MEM] [GRPO step 007/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 007/009] before training inner loop: current=11.95GB  peak_since_reset=15.23GB
[MEM] [GRPO step 007/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=20.94GB
[MEM] [GRPO step 008/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 008/009] before training inner loop: current=11.95GB  peak_since_reset=13.81GB
[MEM] [GRPO step 008/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.31GB
[MEM] [GRPO step 009/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 009/009] before training inner loop: current=11.95GB  peak_since_reset=14.62GB
[MEM] [GRPO step 009/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=19.81GB
```

### Add gradient checkpointing
- commit: `c2bce8d87d8cbffcfbd7534d5328bc3b5fffab55`
- flag: `use_gradient_checkpointing` in `configs/defaults.py`

config: same as above

log:
```
[MEM] after init (model + vLLM + optimizer): current=6.19GB  peak_since_reset=6.19GB
[MEM] [GRPO step 000/009] after rollout generation: current=6.19GB  peak_since_reset=6.22GB
[MEM] [GRPO step 000/009] before training inner loop: current=6.19GB  peak_since_reset=9.49GB
[MEM] [GRPO step 000/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=14.83GB
[MEM] [GRPO step 001/009] after rollout generation: current=11.95GB  peak_since_reset=11.99GB
[MEM] [GRPO step 001/009] before training inner loop: current=11.95GB  peak_since_reset=15.97GB
[MEM] [GRPO step 001/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.57GB
[MEM] [GRPO step 002/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 002/009] before training inner loop: current=11.95GB  peak_since_reset=15.47GB
[MEM] [GRPO step 002/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.12GB
[MEM] [GRPO step 003/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 003/009] before training inner loop: current=11.95GB  peak_since_reset=15.27GB
[MEM] [GRPO step 003/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.93GB
[MEM] [GRPO step 004/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 004/009] before training inner loop: current=11.95GB  peak_since_reset=15.23GB
[MEM] [GRPO step 004/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.90GB
[MEM] [GRPO step 005/009] after rollout generation: current=11.95GB  peak_since_reset=11.99GB
[MEM] [GRPO step 005/009] before training inner loop: current=11.95GB  peak_since_reset=15.47GB
[MEM] [GRPO step 005/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.11GB
[MEM] [GRPO step 006/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 006/009] before training inner loop: current=11.95GB  peak_since_reset=14.98GB
[MEM] [GRPO step 006/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.66GB
[MEM] [GRPO step 007/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 007/009] before training inner loop: current=11.95GB  peak_since_reset=15.19GB
[MEM] [GRPO step 007/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.85GB
[MEM] [GRPO step 008/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 008/009] before training inner loop: current=11.95GB  peak_since_reset=15.28GB
[MEM] [GRPO step 008/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.94GB
[MEM] [GRPO step 009/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 009/009] before training inner loop: current=11.95GB  peak_since_reset=15.57GB
[MEM] [GRPO step 009/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.21GB
```

## Add vLLM sleep mode
- https://docs.vllm.ai/en/latest/features/sleep_mode/#sleep-levels
- flag: `use_vllm_sleep_mode` in `configs/defaults.py`

config: same as above

log:
```
❯ grep '^\[MEM\]' logs/log_vllm_sleep.txt 
[MEM] after init (model + vLLM + optimizer): current=6.19GB  peak_since_reset=6.19GB
[MEM] [GRPO step 000/009] after rollout generation: current=6.19GB  peak_since_reset=6.22GB
[MEM] [GRPO step 000/009] before vLLM sleep: current=6.19GB  peak_since_reset=9.49GB
[MEM] [GRPO step 000/009] after vLLM sleep: current=6.19GB  peak_since_reset=6.19GB
[MEM] [GRPO step 000/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=14.83GB
[MEM] [GRPO step 001/009] after rollout generation: current=11.95GB  peak_since_reset=12.00GB
[MEM] [GRPO step 001/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.96GB
[MEM] [GRPO step 001/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 001/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.57GB
[MEM] [GRPO step 002/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 002/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.47GB
[MEM] [GRPO step 002/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 002/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.12GB
[MEM] [GRPO step 003/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 003/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.27GB
[MEM] [GRPO step 003/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 003/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.93GB
[MEM] [GRPO step 004/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 004/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.27GB
[MEM] [GRPO step 004/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 004/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=17.93GB
[MEM] [GRPO step 005/009] after rollout generation: current=11.95GB  peak_since_reset=11.99GB
[MEM] [GRPO step 005/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.47GB
[MEM] [GRPO step 005/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 005/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.11GB
[MEM] [GRPO step 006/009] after rollout generation: current=11.95GB  peak_since_reset=11.99GB
[MEM] [GRPO step 006/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.46GB
[MEM] [GRPO step 006/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 006/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.11GB
[MEM] [GRPO step 007/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 007/009] before vLLM sleep: current=11.95GB  peak_since_reset=13.55GB
[MEM] [GRPO step 007/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 007/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=16.33GB
[MEM] [GRPO step 008/009] after rollout generation: current=11.95GB  peak_since_reset=11.98GB
[MEM] [GRPO step 008/009] before vLLM sleep: current=11.95GB  peak_since_reset=14.24GB
[MEM] [GRPO step 008/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 008/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=16.97GB
[MEM] [GRPO step 009/009] after rollout generation: current=11.95GB  peak_since_reset=11.99GB
[MEM] [GRPO step 009/009] before vLLM sleep: current=11.95GB  peak_since_reset=15.57GB
[MEM] [GRPO step 009/009] after vLLM sleep: current=11.95GB  peak_since_reset=11.95GB
[MEM] [GRPO step 009/009] after training inner loop (peak = training VRAM): current=11.95GB  peak_since_reset=18.21GB
```
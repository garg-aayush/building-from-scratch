from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from omegaconf import OmegaConf


@dataclass
class PathConfig:
    train_data_file: Path = Path("/home/aayush/DATA/GRPO/train.jsonl")           # jsonl file containing the training data
    val_data_file: Path = Path("/home/aayush/DATA/GRPO/validation.jsonl")        # jsonl file containing the validation data
    prompt_template_file: Path = Path("/home/aayush/DATA/GRPO/r1_zero.prompt")   # prompt template file
    model_path: Path = Path("/home/aayush/DATA/Qwen2.5-Math-1.5B")               # path to the qwen2.5-math-1.5b model checkpoint
    output_dir: Path = Path("/home/aayush/RESULTS/GRPO/run")                     # directory to save the results

@dataclass
class TrainingConfig:
    # common
    seed: int = 1337
    device: str = "cuda:0"
    dtype: str = "bfloat16"                          # "float16" or "bfloat16"
    
    # model
    attention_type: str = "flash_attention_2"
    use_compile: bool = False                        # whether to use torch.compile for the model
    
    # AdamW optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int = 4
    max_tokens: int = 1024
    stop: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    
    # GRPO parameters
    n_grpo_steps: int = 200                             # number of GRPO steps
    advantage_eps: float = 1e-6                         # epsilon for advantage normalization
    rollout_batch_size: int = 256                       # number of rollouts per batch
    group_size: int = 8                                 # size of each group
    epochs_per_rollout_batch: int = 1                   # On-policy (off-policy if > 1)
    train_batch_size: int = 256                         # On-policy
    gradient_accumulation_steps: int = 128              # microbatch size is 2
    loss_type: str = "no_baseline"                      # "no_baseline", "reinforce_with_baseline", "grpo_no_clip", "grpo_clip"
    use_std_normalization: bool = True                  # whether to use standard normalization for advantages
    
@dataclass
class VllmConfig:
    gpu_memory_utilization: float = 0.2                 # fraction of GPU memory to reserve for vLLM
    dtype: str = "bfloat16"                             # dtype for vLLM
    enable_prefix_caching: bool = True                  # whether to enable prefix caching for vLLM
    
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from omegaconf import OmegaConf


@dataclass
class PathConfig:
    train_data_file: Path = field(default_factory=lambda: Path("/home/aayush/DATA/GRPO/train.jsonl"))           # jsonl file containing the training data
    val_data_file: Path = field(default_factory=lambda: Path("/home/aayush/DATA/GRPO/validation.jsonl"))        # jsonl file containing the validation data
    prompt_template_file: Path = field(default_factory=lambda: Path("/home/aayush/DATA/GRPO/r1_zero.prompt"))   # prompt template file
    model_path: Path = field(default_factory=lambda: Path("/home/aayush/DATA/Qwen2.5-Math-1.5B"))               # path to the qwen2.5-math-1.5b model checkpoint
    output_dir: Path = field(default_factory=lambda: Path("/home/aayush/RESULTS/GRPO/run"))                     # directory to save the results

@dataclass
class TrainingConfig:
    # common
    seed: int = field(default_factory=lambda: 1337)
    device: str = field(default_factory=lambda: "cuda:0")
    dtype: str = field(default_factory=lambda: "bfloat16")                          # "float16" or "bfloat16"
    
    # model
    attention_type: str = field(default_factory=lambda: "flash_attention_2")
    use_compile: bool = field(default_factory=lambda: False)                        # whether to use torch.compile for the model
    
    # AdamW optimizer
    learning_rate: float = field(default_factory=lambda: 1e-5)
    weight_decay: float = field(default_factory=lambda: 0.0)
    adam_beta1: float = field(default_factory=lambda: 0.9)
    adam_beta2: float = field(default_factory=lambda: 0.95)
    adam_eps: float = field(default_factory=lambda: 1e-8)
    
    # sampling parameters
    temperature: float = field(default_factory=lambda: 1.0)
    top_p: float = field(default_factory=lambda: 1.0)
    min_tokens: int = field(default_factory=lambda: 4)
    max_tokens: int = field(default_factory=lambda: 1024)
    stop: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = field(default_factory=lambda: True)
    
@dataclass
class VllmConfig:
    gpu_memory_utilization: float = field(default_factory=lambda: 0.2)          # fraction of GPU memory to reserve for vLLM
    dtype: str = field(default_factory=lambda: "bfloat16")
    enable_prefix_caching: bool = field(default_factory=lambda: True)    # whether to enable prefix caching for vLLM
    
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
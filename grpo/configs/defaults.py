from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class PathConfig:
    train_data_file: Path = field(default_factory=lambda: Path("/home/aayush/DATA/GRPO/train.jsonl"))           # jsonl file containing the training data
    val_data_file: Path = field(default_factory=lambda: Path("/home/aayush/DATA/GRPO/validation.jsonl"))        # jsonl file containing the validation data
    prompt_template_file: Path = field(default_factory=lambda: Path("/home/aayush/DATA/GRPO/r1_zero.prompt"))   # prompt template file
    model_path: Path = field(default_factory=lambda: Path("/home/aayush/DATA/Qwen2.5-Math-1.5B"))               # path to the qwen2.5-math-1.5b model checkpoint
    output_dir: Path = field(default_factory=lambda: Path("/home/aayush/RESULTS/GRPO/run"))                     # directory to save the results

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
import json
import random
from typing import Callable, List

import numpy as np
import torch
from configs.defaults import Config
from utils.constants import DTYPE_MAPPING
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def set_seed(seed: int):
    """
    Set the seed for the random number generators.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------#
# Pretty print the input_config
# -------------------------------------------------------------#
def pretty_print(input_config: dict | list | str | None, title: str | None = None, is_sub_title: bool = False) -> None:
    """
    Pretty print the input_config.
    """
    if title is not None:
        if is_sub_title:
            print(f"{'-'*30}\n{title}:\n{'-'*30}")
        else:
            print("="*25 + f" {title} " + "="*25)
    if isinstance(input_config, dict):
        for k,v in input_config.items():
            if isinstance(v, dict):
                print(f"{k:<25}:")
                for kk, vv in v.items():
                    print(f"    {kk:<25}: {vv}")
            else:
                print(f"{k:<25}: {v}")
    elif isinstance(input_config, list):
        for i, v in enumerate(input_config):
            print(f"{i:<25}: {v}")
    elif isinstance(input_config, str):
        print(input_config)
    elif input_config is None:
        pass
    else:
        raise ValueError(f"Unsupported type: {type(input_config)}")


# -------------------------------------------------------------#
# functions to initialize the vLLM model and load the policy into the vLLM model during training
# -------------------------------------------------------------#
def init_vllm(seed: int, cfg: Config):
    """
    Start the inference process, here we use vLLM to hold a model on
    the same GPU as the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can place the vLLM model on the desired device
    # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # Note: Removed profiling_patch as it's vLLM version-dependent and causes AttributeError in newer versions
    # with world_size_patch:
    vllm_init_params = {
        "model": cfg.paths.model_path.as_posix(),
        "gpu_memory_utilization": cfg.vllm.gpu_memory_utilization,
        "dtype": DTYPE_MAPPING[cfg.vllm.dtype],
        "enable_prefix_caching": cfg.vllm.enable_prefix_caching,
    }
    pretty_print(vllm_init_params, title="vLLM model initialization parameters")
    return LLM(**vllm_init_params)

# -------------------------------------------------------------#
# functions to load the dataset
# -------------------------------------------------------------#
def load_dataset(data_file: str, data_type: str='train', prompt_template: str=None):
    with open(data_file, 'r') as f:
        if data_type in ['train', 'val']:
            return [json.loads(line) for line in f]
        elif data_type == 'prompt':
            return f.read()
        else:
            raise ValueError(f"Invalid data type: {data_type}")
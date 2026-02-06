import json
import random
from typing import Callable, List

import numpy as np
import torch


# -------------------------------------------------------------#
# functions to set the seed
# -------------------------------------------------------------#
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
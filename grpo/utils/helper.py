import json
import random
from typing import Callable, List


# -------------------------------------------------------------#
# Pretty print the input_config
# -------------------------------------------------------------#
def pretty_print(input_config: dict | list | str, title: str = "Config", is_super_title: bool = False) -> None:
    """
    Pretty print the input_config.
    """
    if is_super_title:
        print(f"{'-'*100}\n{title}:\n{'-'*100}")
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
    else:
        raise ValueError(f"Unsupported type: {type(input_config)}")
# -------------------------------------------------------------#
# Pretty print the input_config
# -------------------------------------------------------------#
def pretty_print(input_config: dict | list | str, title: str = "Config") -> None:
    """
    Pretty print the input_config.
    """
    print(f"{'-'*100}\n{title}:\n{'-'*100}")
    if isinstance(input_config, dict):
        for k,v in input_config.items():
            if isinstance(v, dict):
                print(f"{k:<20}:")
                for kk, vv in v.items():
                    print(f"    {kk:<20}: {vv}")
            else:
                print(f"{k:<20}: {v}")
    elif isinstance(input_config, list):
        for i, v in enumerate(input_config):
            print(f"{i:<20}: {v}")
    elif isinstance(input_config, str):
        print(input_config)
    else:
        raise ValueError(f"Unsupported type: {type(input_config)}")
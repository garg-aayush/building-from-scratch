from configs.defaults import Config
from omegaconf import OmegaConf
from utils.helper import pretty_print

default_config = OmegaConf.structured(Config)

# print the config
config_dict = OmegaConf.to_container(default_config, resolve=True)
pretty_print(config_dict, title="Config")
del config_dict

import inspect

import torch
from configs.defaults import Config
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEFAULT_CONFIG, DTYPE_MAPPING
from utils.helper import init_vllm, load_dataset, pretty_print, set_seed

config_path = "/home/aayush/repos/building-from-scratch/grpo/configs/dummy.yaml"

# load the config
if config_path is not None:
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(DEFAULT_CONFIG, config)
else:
    config = DEFAULT_CONFIG
config_dict = OmegaConf.to_container(config, resolve=True)
pretty_print(config_dict, title="Config")
del config_dict

# -------------------------------------------------------------#
# Seed and precision setup
# -------------------------------------------------------------#
pretty_print(f"Setting the seed to {config.training.seed} and using tf32 precision...", title="Set Random Seed")
set_seed(config.training.seed)
torch.set_float32_matmul_precision("high") # use tf32

# -------------------------------------------------------------#
# Load train and val dataset
# -------------------------------------------------------------#
pretty_print(None, title="Load datasets")
# prompt template
pretty_print(f"Loading prompt template from {config.paths.prompt_template_file}...")
prompt_template = load_dataset(data_file=config.paths.prompt_template_file, data_type='prompt')
pretty_print(prompt_template, title="Prompt template", is_sub_title=True)
# train dataset
pretty_print(f"Loading train dataset from {config.paths.train_data_file}...")
train_dataset = load_dataset(data_file=config.paths.train_data_file, data_type='train')
pretty_print(f"Train dataset size: {len(train_dataset)}", title="Train dataset", is_sub_title=True)
pretty_print(train_dataset[:5])
# val dataset
pretty_print(f"Loading val dataset from {config.paths.val_data_file}...")
val_dataset = load_dataset(data_file=config.paths.val_data_file, data_type='val')
pretty_print(f"Val dataset size: {len(val_dataset)}", title="Val dataset", is_sub_title=True)
pretty_print(val_dataset[:5])

# -------------------------------------------------------------#
# Initialize the vLLM model
# -------------------------------------------------------------#
pretty_print("Initializing the vLLM model...", title="vLLM model initialization")
# vllm_model = init_vllm(config.training.seed, config)


# -------------------------------------------------------------#
# Initialize the tokenizer and model
# -------------------------------------------------------------#
pretty_print(None, title="Tokenizer and model initialization")
# tokenizer
pretty_print(f"Initializing the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.paths.model_path.as_posix())
# model
pretty_print(f"Initializing the model...")
model = AutoModelForCausalLM.from_pretrained(
            config.paths.model_path.as_posix(),
            dtype=DTYPE_MAPPING[config.training.dtype], 
            attn_implementation=config.training.attention_type, 
            device_map=config.training.device
        )
# compile the model
if config.training.use_compile:
    print(f"compile flag: {config.training.use_compile}, compiling the model...")
    model = torch.compile(model)
else:
    print(f"compile flag: {config.training.use_compile}, skipping model compilation...")


# -------------------------------------------------------------#
# Setup the optimizer
# -------------------------------------------------------------#
pretty_print("Setup the AdamW optimizer...", title="AdamW optimizer setup")
# check if fused AdamW is available
use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and config.training.device.startswith("cuda")
print(f"Using fused AdamW: {use_fused}")
# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    betas=(config.training.adam_beta1, config.training.adam_beta2),
    eps=config.training.adam_eps,
    fused=use_fused,
)
print(optimizer)
import inspect
import random

import torch
from configs.defaults import Config
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEFAULT_CONFIG, DTYPE_MAPPING
from utils.dataset import load_dataset, tokenize_prompt_and_output
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.grpo import compute_group_normalized_rewards
from utils.helper import pretty_print, set_seed
from utils.vllm import init_vllm
from vllm import SamplingParams

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
vllm_model = init_vllm(config.training.seed, config)


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


# -------------------------------------------------------------#
# Training loop
# -------------------------------------------------------------#
pretty_print("Starting the training loop...", title="GRPO Training loop")
# number of prompts per rollout batch
n_prompts_per_rollout_batch = config.training.rollout_batch_size // config.training.group_size
pretty_print(f"Number of prompts per rollout batch: {n_prompts_per_rollout_batch}")

# sampling params
sampling_params = SamplingParams(
    temperature=config.training.temperature,
    top_p=config.training.top_p,
    max_tokens=config.training.max_tokens,
    stop=[str(s) for s in config.training.stop],
    include_stop_str_in_output=config.training.include_stop_str_in_output,
    min_tokens=config.training.min_tokens
)
# reward function
reward_fn = r1_zero_reward_fn

# GRPO loop
for grpo_step in range(config.training.n_grpo_steps):
    # get a random batch of prompts
    cur_batch = random.sample(train_dataset, n_prompts_per_rollout_batch)
    
    # create a list of repeated prompts and ground truths
    rep_rollout_prompts = [prompt_template.replace("{question}", ex['problem']) for ex in cur_batch 
                           for _ in range(config.training.group_size)]
    rep_rollout_ground_truths = [ex['answer'] for ex in cur_batch for _ in range(config.training.group_size)]
    
    # generate rollouts
    rollout_outputs = vllm_model.generate(rep_rollout_prompts, sampling_params)
    rollout_responses = [output.outputs[0].text for output in rollout_outputs]
    
    # compute rewards
    rollout_advantages, rollout_raw_rewards, rollout_rewards_meta = compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        rep_rollout_ground_truths,
        config.training.group_size,
        config.training.advantage_eps,
        config.training.use_std_normalization,
    )
    
    # print some random rollout responses
    pretty_print(None, title="Random rollout responses")
    for i in random.sample(range(len(rollout_responses)), 5):
        pretty_print(None, title=f"Rollout {i}", is_sub_title=True)
        pretty_print(f"Prompt -> {rep_rollout_prompts[i]}")
        pretty_print(f"Response -> {rollout_responses[i]}")
        pretty_print(f"Ground truth -> {rep_rollout_ground_truths[i]}")
        pretty_print(f"Advantage -> {rollout_advantages[i]}")
        pretty_print(f"Raw reward -> {rollout_raw_rewards[i]}")
    
    # tokenize the rollout_responses
    tokenized_train_data = tokenize_prompt_and_output(rep_rollout_prompts, rollout_responses, tokenizer)
    pretty_print(tokenized_train_data, title="Tokenized train data", is_sub_title=True)
    
    
    # # compute old_log_probs over full rollout_batch_size
    # num_train_steps_per_epoch = config.training.rollout_batch_size // confg.training.train_batch_size
    # pretty_print(f"Num. train steps/epoch: {num_train_steps_per_epoch}, train bz: {config.training.train_batch_size}, rollout bz: {config.training.rollout_batch_size}")
    # model.eval()
    # with torch.no_grad():
    
    exit()
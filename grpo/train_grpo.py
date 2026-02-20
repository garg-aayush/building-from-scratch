import os

# disable v1 multiprocessing to avoid 'LLMEngine' object has no attribute 'model_executor' error in vLLM 0.11.0
# otherwise downgrade vllm to 0.10.2
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 

import inspect
import random
import time

import torch
from configs.defaults import Config
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEFAULT_CONFIG, DTYPE_MAPPING
from utils.dataset import load_dataset, tokenize_prompt_and_output
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.grpo import (compute_group_normalized_rewards,
                        get_response_log_probs, grpo_microbatch_train_step)
from utils.helper import log_memory, pretty_print, set_seed
from utils.vllm import init_vllm, load_policy_into_vllm_instance
from vllm import SamplingParams

# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 


config_path = "/home/aayush/repos/building-from-scratch/grpo/configs/dummy.yaml"


# -------------------------------------------------------------#
# Load the config
# -------------------------------------------------------------#
# load the config
if config_path is not None:
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(DEFAULT_CONFIG, config)
else:
    config = DEFAULT_CONFIG
config_dict = OmegaConf.to_container(config, resolve=True)
pretty_print(config_dict, title="Config")
del config_dict

assert config.training.train_batch_size % config.training.gradient_accumulation_steps == 0, f"train_batch_size must be divisible by gradient_accumulation_steps"
micro_train_batch_size = config.training.train_batch_size // config.training.gradient_accumulation_steps
pretty_print(f"Micro train batch size: {micro_train_batch_size}")
assert config.training.rollout_batch_size % config.training.group_size == 0, f"rollout_batch_size must be divisible by group_size"
n_prompts_per_rollout_batch = config.training.rollout_batch_size // config.training.group_size
pretty_print(f"Number of prompts per rollout batch: {n_prompts_per_rollout_batch}")
assert config.training.train_batch_size >= config.training.group_size, f"train_batch_size must be greater than or equal to group_size"

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

if config.training.track_peak_memory:
    log_memory("after init (model + vLLM + optimizer)", config.training.device, reset_after=True)


# -------------------------------------------------------------#
# Training loop
# -------------------------------------------------------------#
pretty_print("Starting the training loop...", title="GRPO Training loop")

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
    grpo_step_title = f"GRPO step {grpo_step:03d}/{config.training.n_grpo_steps-1:03d}"
    pretty_print(None, title=grpo_step_title)
    
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
    
    if config.training.track_peak_memory:
        log_memory(f"[{grpo_step_title}] after rollout generation", config.training.device, reset_after=True)

    # tokenize the rollout_responses
    tokenized_train_data = tokenize_prompt_and_output(rep_rollout_prompts, rollout_responses, tokenizer)
    pretty_print(tokenized_train_data, title="Tokenized train data", is_sub_title=True)
    
    # compute old_log_probs over full rollout_batch_size
    old_log_probs = None
    if config.training.loss_type in ["grpo_clip", "grpo_no_clip"]:
        pretty_print("Computing old_log_probs over full rollout_batch_size...", title=f"{grpo_step_title} - Old log probs computation", is_sub_title=True)
        model.eval()
        old_log_probs = []
        # compute old log probs for each microbatch
        total_train_size, batch_size = len(tokenized_train_data["input_ids"]), config.training.old_log_probs_train_size
        for idx in range(0, len(tokenized_train_data["input_ids"]), config.training.old_log_probs_train_size):
            input_ids, labels = map(lambda x: x[idx:idx+batch_size].to(config.training.device), [tokenized_train_data["input_ids"], tokenized_train_data["labels"]])
            with torch.no_grad():
                with torch.autocast(device_type=config.training.device.split(":")[0], dtype=DTYPE_MAPPING[config.training.dtype]):
                    log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
                old_log_probs.append(log_probs)
        # concatenate the old log probs for each microbatch and offload to CPU
        old_log_probs = torch.cat(old_log_probs, dim=0).cpu()
        old_log_probs_mem_mb = old_log_probs.element_size() * old_log_probs.nelement() / 1024 ** 2
        pretty_print(f"Old log probs shape: {old_log_probs.shape}, memory: {old_log_probs_mem_mb:.2f} MB")
        # delete unnecessary variables
        del log_probs, input_ids, labels, total_train_size, batch_size, old_log_probs_mem_mb
    
    # clear torch cache (to save memory)
    torch.cuda.empty_cache()
    if config.training.track_peak_memory:
        log_memory(f"[{grpo_step_title}] before training inner loop", config.training.device, reset_after=True)
    model.train()

    # Inner loop: grpo over the rollout batch
    for train_epoch in range(config.training.epochs_per_rollout_batch):
        pretty_print(f"", title=f"{grpo_step_title} - GRPO epoch {train_epoch:02d}/{config.training.epochs_per_rollout_batch-1:02d}", is_sub_title=True)
        
        # loop through train steps
        num_train_steps = config.training.rollout_batch_size // config.training.train_batch_size
        for train_step in range(num_train_steps):
            pretty_print(f"", title=f"{grpo_step_title} - GRPO inner step {train_step:02d}/{num_train_steps-1:02d}", is_sub_title=True)
            
            loss_accum = 0.0
            entropy_accum = 0.0
            total_response_tokens = 0
            start_time = time.time()
            # loop through microbatches
            for idx in tqdm(range(config.training.gradient_accumulation_steps), desc="Microbatches", leave=False):
                
                # get the base index, start index, and end index for the microbatch
                base_idx = train_step * config.training.train_batch_size
                start_idx = base_idx + idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                # pretty_print(f"microbatch idx: {idx:03d}, base_idx: {base_idx:03d}, start_idx: {start_idx:03d}, end_idx: {end_idx:03d}")
                
                # microbatch of input_ids, labels, and response_mask
                microbatch = {k: v[start_idx:end_idx].to(config.training.device) for k, v in tokenized_train_data.items()}
                
                # compute current policy log_probs and token_entropy for the microbatch
                with torch.autocast(device_type=config.training.device.split(":")[0], dtype=DTYPE_MAPPING[config.training.dtype]):
                    cur_log_probs_result = get_response_log_probs(model, microbatch["input_ids"], microbatch["labels"], True)
                
                # calculate the loss for the microbatch
                current_log_probs = cur_log_probs_result["log_probs"]
                token_entropy = cur_log_probs_result["token_entropy"]
                old_log_probs_microbatch = old_log_probs[start_idx:end_idx].to(config.training.device) if old_log_probs is not None else None
                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=current_log_probs,
                    response_mask=microbatch["response_mask"],
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                    loss_type=config.training.loss_type,
                    raw_rewards=rollout_raw_rewards[start_idx:end_idx].unsqueeze(-1).to(config.training.device),
                    advantages=rollout_advantages[start_idx:end_idx].unsqueeze(-1).to(config.training.device),
                    old_log_probs=old_log_probs_microbatch,
                    cliprange=config.training.cliprange
                )
                
                # accumulate
                loss_accum += loss.detach()
                entropy_accum += (token_entropy * microbatch["response_mask"]).sum().detach()
                total_response_tokens += microbatch["response_mask"].sum().detach()
                del microbatch, cur_log_probs_result, current_log_probs, token_entropy, loss, old_log_probs_microbatch
            
            
            avg_entropy = entropy_accum / total_response_tokens
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            # update the model parameters
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            dt = time.time() - start_time
            
            # calculate mean rewards for the rollout batch
            step_mean_format_reward = 0.0
            step_mean_answer_reward = 0.0
            step_mean_reward = 0.0
            rewards_len = len(rollout_rewards_meta['rewards'])
            for r in rollout_rewards_meta['rewards']:
                step_mean_format_reward += r["format_reward"] / rewards_len
                step_mean_answer_reward += r["answer_reward"] / rewards_len
                step_mean_reward += r["reward"] / rewards_len
            
            # print the step metrics
            pretty_print(f"grpo_step: {grpo_step:03d} | train_step: {train_step:02d} | loss: {loss_accum.item():.4f} | entropy: {avg_entropy:.4f} | reward: {step_mean_reward:.4f} | answer_reward: {step_mean_answer_reward:.4f} | format_reward: {step_mean_format_reward:.4f} | grad_norm: {grad_norm:.4f} | lr: {config.training.learning_rate:.6f} | time: {dt:.2f}s")
            

            torch.cuda.empty_cache()

    if config.training.track_peak_memory:
        log_memory(f"[{grpo_step_title}] after training inner loop (peak = training VRAM)", config.training.device, reset_after=True)

    # load the model weights
    load_policy_into_vllm_instance(model, vllm_model)
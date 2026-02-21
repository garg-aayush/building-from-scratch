import os

# disable v1 multiprocessing to avoid 'LLMEngine' object has no attribute 'model_executor' error in vLLM 0.11.0
# otherwise downgrade vllm to 0.10.2
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import inspect
import json
import random
import time

import torch
import wandb
from configs.defaults import Config
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEFAULT_CONFIG, DTYPE_MAPPING
from utils.dataset import load_dataset, tokenize_prompt_and_output
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.evaluate import evaluate_vllm
from utils.grpo import (compute_group_normalized_rewards,
                        get_response_log_probs, grpo_microbatch_train_step)
from utils.helper import log_memory, pretty_print, set_seed
from utils.vllm import init_vllm, load_policy_into_vllm_instance
from vllm import SamplingParams

if __name__ == "__main__":

    # -------------------------------------------------------------#
    # Load the config
    # -------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (overrides defaults)")
    args = parser.parse_args()

    if args.config is not None:
        config = OmegaConf.merge(DEFAULT_CONFIG, OmegaConf.load(args.config))
    else:
        config = DEFAULT_CONFIG
    config_dict = OmegaConf.to_container(config, resolve=True)
    pretty_print(config_dict, title="Config")

    use_wandb = bool(config.training.wandb_project)
    if use_wandb:
        pretty_print("Initializing wandb...", title="Wandb initialization")
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name or None,
            config=config_dict,
            tags=config.training.wandb_tags,
        )
        wandb.define_metric("grpo_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="grpo_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
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
    # gradient checkpointing: trade compute for activation memory
    if config.training.use_gradient_checkpointing:
        pretty_print("Gradient checkpointing enabled...")
        model.gradient_checkpointing_enable()
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
    if config.training.use_bnb_adamw8bit:
        import bitsandbytes as bnb
        pretty_print("Using bitsandbytes AdamW8bit optimizer.")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
        )
    else:
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and config.training.device.startswith("cuda")
        pretty_print(f"Using torch AdamW (fused={use_fused}).")
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

    # step counter for wandb eval x-axis
    eval_step = 0

    # evaluation before any training
    if config.training.eval_interval > 0:
        pretty_print("Running base model evaluation (pre-training)...", title="Base Model Evaluation")
        eval_metrics = evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=reward_fn,
            val_dataset=val_dataset,
            prompt_template=prompt_template,
            sampling_params=sampling_params,
            max_val_examples=config.training.max_val_examples,
        )
        pretty_print(f"[EVAL] grpo_step: 000 | n_examples: {eval_metrics['n_examples']} | reward: {eval_metrics['mean_reward']:.4f} | answer_reward: {eval_metrics['mean_answer_reward']:.4f} | format_reward: {eval_metrics['mean_format_reward']:.4f}")
        if use_wandb:
            wandb.log({
                "eval/reward": eval_metrics['mean_reward'],
                "eval/answer_reward": eval_metrics['mean_answer_reward'],
                "eval/format_reward": eval_metrics['mean_format_reward'],
                "eval_step": eval_step,
            })
            eval_step += 1

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

        # save a random sample of rollouts to disk
        if config.training.n_rollouts_to_log > 0:
            save_indices = random.sample(range(len(rollout_responses)), min(config.training.n_rollouts_to_log, len(rollout_responses)))
            rollouts_dir = config.paths.output_dir / "rollouts"
            rollouts_dir.mkdir(parents=True, exist_ok=True)
            rollout_records = [
                {
                    "prompt": rep_rollout_prompts[i],
                    "response": rollout_responses[i],
                    "ground_truth": rep_rollout_ground_truths[i],
                    "advantage": rollout_advantages[i].item(),
                    "reward": rollout_rewards_meta["rewards"][i]["reward"],
                    "format_reward": rollout_rewards_meta["rewards"][i]["format_reward"],
                    "answer_reward": rollout_rewards_meta["rewards"][i]["answer_reward"],
                }
                for i in save_indices
            ]
            rollout_file = rollouts_dir / f"rollouts_step_{grpo_step:03d}.jsonl"
            with open(rollout_file, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rollout_records) + "\n")
            pretty_print(f"Saved {len(rollout_records)} rollouts to {rollout_file}")

        if config.training.track_peak_memory:
            log_memory(f"[{grpo_step_title}] after rollout generation", config.training.device, reset_after=True)

        # tokenize the rollout_responses
        tokenized_train_data = tokenize_prompt_and_output(rep_rollout_prompts, rollout_responses, tokenizer)
        pretty_print(tokenized_train_data, title="Tokenized train data", is_sub_title=True)
        mean_response_length = tokenized_train_data["response_mask"].sum(dim=1).float().mean().item()

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

        # sleep vLLM to free its GPU memory (weights + KV cache) during training
        if config.training.use_vllm_sleep_mode:
            pretty_print("Sleeping vLLM to free its GPU memory (weights + KV cache) during training...")
            vllm_model.sleep(level=1)
        if config.training.track_peak_memory:
            log_memory(f"[{grpo_step_title}] after vLLM sleep", config.training.device, reset_after=True)

        # clear torch cache (to save memory)
        torch.cuda.empty_cache()
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
                clip_fraction_accum = 0.0
                mean_ratio_accum = 0.0
                total_response_tokens = 0
                start_time = time.time()
                # loop through microbatches
                for idx in tqdm(range(config.training.gradient_accumulation_steps), desc="Microbatches", leave=False):

                    # get the base index, start index, and end index for the microbatch
                    base_idx = train_step * config.training.train_batch_size
                    start_idx = base_idx + idx * micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size

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
                        cliprange=config.training.cliprange,
                        norm_mode=config.training.normalize_mode,
                        norm_constant=config.training.normalize_constant,
                    )

                    # accumulate
                    loss_accum += loss.detach()
                    entropy_accum += (token_entropy * microbatch["response_mask"]).sum().detach()
                    total_response_tokens += microbatch["response_mask"].sum().detach()
                    clip_fraction_accum += meta.get("clip_fraction", 0.0)
                    mean_ratio_accum += meta.get("mean_ratio", 0.0)
                    del microbatch, cur_log_probs_result, current_log_probs, token_entropy, loss, old_log_probs_microbatch


                avg_entropy = entropy_accum / total_response_tokens
                avg_clip_fraction = clip_fraction_accum / config.training.gradient_accumulation_steps
                avg_mean_ratio = mean_ratio_accum / config.training.gradient_accumulation_steps
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
                pretty_print(f"[TRAIN] grpo_step: {grpo_step:03d} | loss: {loss_accum.item():.4f} | entropy: {avg_entropy:.4f} | reward: {step_mean_reward:.4f} | answer_reward: {step_mean_answer_reward:.4f} | format_reward: {step_mean_format_reward:.4f} | clip_frac: {avg_clip_fraction:.4f} | mean_ratio: {avg_mean_ratio:.4f} | grad_norm: {grad_norm:.4f} | lr: {config.training.learning_rate:.6f} | mean_response_len: {mean_response_length:.1f} | time: {dt:.2f}s")
                if use_wandb:
                    wandb.log({
                        "train/loss": loss_accum.item(),
                        "train/entropy": avg_entropy.item(),
                        "train/reward": step_mean_reward,
                        "train/answer_reward": step_mean_answer_reward,
                        "train/format_reward": step_mean_format_reward,
                        "train/clip_fraction": avg_clip_fraction,
                        "train/mean_ratio": avg_mean_ratio,
                        "train/grad_norm": grad_norm.item(),
                        "train/mean_response_length": mean_response_length,
                        "grpo_step": grpo_step,
                    })

                torch.cuda.empty_cache()

        if config.training.track_peak_memory:
            log_memory(f"[{grpo_step_title}] after training inner loop (peak = training VRAM)", config.training.device, reset_after=True)
        # wake vLLM before loading updated policy weights and next generation step
        if config.training.use_vllm_sleep_mode:
            vllm_model.wake_up()
        # load the model weights
        load_policy_into_vllm_instance(model, vllm_model)

        # intermediate evaluation on a subset of val_dataset
        is_last_step = grpo_step == config.training.n_grpo_steps - 1
        if config.training.eval_interval > 0 and ((grpo_step+1) % config.training.eval_interval == 0 or is_last_step):
            pretty_print(f"Running intermediate evaluation on {config.training.max_val_examples} val examples...", title=f"{grpo_step_title} - Intermediate Evaluation", is_sub_title=True)
            eval_metrics = evaluate_vllm(
                vllm_model=vllm_model,
                reward_fn=reward_fn,
                val_dataset=val_dataset,
                prompt_template=prompt_template,
                sampling_params=sampling_params,
                max_val_examples=config.training.max_val_examples,
            )
            pretty_print(f"[EVAL] grpo_step: {grpo_step:03d} | n_examples: {eval_metrics['n_examples']} | reward: {eval_metrics['mean_reward']:.4f} | answer_reward: {eval_metrics['mean_answer_reward']:.4f} | format_reward: {eval_metrics['mean_format_reward']:.4f}")
            if use_wandb:
                wandb.log({
                    "eval/reward": eval_metrics['mean_reward'],
                    "eval/answer_reward": eval_metrics['mean_answer_reward'],
                    "eval/format_reward": eval_metrics['mean_format_reward'],
                    "eval_step": eval_step,
                    "grpo_step": grpo_step,
                })
                eval_step += 1

        # checkpoint saving
        if config.training.checkpoint_interval > 0 and ((grpo_step+1) % config.training.checkpoint_interval == 0 or is_last_step):
            ckpt_dir = config.paths.output_dir / f"checkpoint_{grpo_step:03d}"
            pretty_print(f"Saving checkpoint to {ckpt_dir}...", title=f"{grpo_step_title} - Checkpoint", is_sub_title=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    if use_wandb:
        wandb.finish()

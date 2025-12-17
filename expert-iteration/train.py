import os

# disable v1 multiprocessing to avoid 'LLMEngine' object has no attribute 'model_executor' error in vLLM 0.11.0
# otherwise downgrade vllm to 0.10.2
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
import inspect
import json
import random
import time
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from utils.dataloader import SftTrainDataLoaderLite
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.helper_fns import (evaluate_vllm, init_vllm,
                              load_policy_into_vllm_instance, prepare_val_data,
                              pretty_print, create_ei_filtered_data)
from utils.post_train import (get_response_log_probs, sft_eval_step,
                              sft_microbatch_train_step,
                              tokenize_prompt_and_output)
from vllm import LLM, SamplingParams
import argparse

# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
# wandb tracking setup
seed = 1337
wandb_project = "expert-iter"

# Model config
model_name = "/home/qwen/"
# model_name = "/home/RESULTS/qwen-2.5-math-sft-filtered/checkpoint_38"
dtype = "bfloat16"  # "float16" or "bfloat16"
attention_type = "flash_attention_2"
use_compile = True

# Device & vLLM config
device = "cuda:0"  # please use a GPU for training
device_type = "cuda" if device.startswith("cuda") else "cpu"

# train and val data
train_data_file = "/home/DATA/EXPERT-ITER/sft_gpt-oss-120b_filtered.jsonl"
prompt_template_file = "/home/DATA/EXPERT-ITER/r1_zero.prompt"
val_data_file = "/home/DATA/EXPERT-ITER/val.jsonl"
tmp_dir = "/tmp"
use_generated_reasoning = True

# Training hyperparameters
total_batch_size_list = [8, 32, 64]
# learning_rate_list = [1.7e-5, 3.53e-5, 5e-5] # 5e-5 max lr and follow sqrt(2) rule for learning rate scaling
learning_rate_list = [3.5e-5, 5e-5, 7e-5] # 1e-4 for batch size of 128 and follow sqrt(2) rule for learning rate scaling 
batch_boundary_list = [24, 128]
micro_batch_size = 2
val_batch_size = 4
grad_norm_clip = 1.0
use_per_token_loss = True # use per-token loss instead of per-sequence loss
normalize_constant = 1.0 # normalization constant for the loss
batch_per_ei = 512
num_ei = 5
num_rollouts = 2 # number of outputs to generate for each example
wandb_run_name = f"run_D{num_ei}_R{num_rollouts}"
eval_ei_steps = [1, 2, 3, 4, 5]
save_ei_steps = [3, 5]
num_epochs = 1

# Checkpointing & logging
output_dir = f"/home/RESULTS/{wandb_run_name}"
run_intermediate_eval = True # run intermediate evaluation on the validation set
max_val_examples = 6000  # maximum number of validation examples to evaluate on

# vLLM config
vllm_dtype = "bfloat16"  # vLLM expects string, not torch dtype
vllm_gpu_memory_utilization = 0.2  # reserve 20% of the GPU memory
vllm_max_model_len = 2048  # maximum model length
vllm_max_num_seqs = 128  # maximum number of sequences
vllm_sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "min_tokens": 4,
    "max_tokens": 1024,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
    "n": num_rollouts
}
vllm_sampling_params_eval = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True
}

# -------------------------------------------------------------#
# Helper functions
# -------------------------------------------------------------#
def get_lr_and_max_steps(num_examples, boundary_list, bs_list, lr_list, data_loader, micro_bs, num_epochs):
    tot_examples = len(data_loader) * num_epochs
    if num_examples < boundary_list[0]:
        lr, bs = lr_list[0], bs_list[0]
    elif num_examples < boundary_list[1]:
        lr, bs = lr_list[1], bs_list[1]
    else:
        lr, bs = lr_list[2], bs_list[2]
    max_steps = max(tot_examples // bs, 1)
    cur_grad_steps = min(tot_examples, bs) // micro_bs
    print(f"num_examples: {num_examples}\nnum_epochs: {num_epochs}\nbs: {bs}\nmax_steps: {max_steps}\ncur_grad_steps: {cur_grad_steps}\nlearning_rate: {lr:.4e}")
    return lr, max_steps, cur_grad_steps

# -------------------------------------------------------------#
# Seed and precision setup
# -------------------------------------------------------------#
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# use tf32
torch.set_float32_matmul_precision("high")


# -------------------------------------------------------------#
# Main function
# -------------------------------------------------------------#
if __name__ == '__main__':
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_per_ei", type=int, default=512)
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--wandb_run_name", type=str, default="run_G2_R2")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="/home/qwen/")

    args = parser.parse_args()
    batch_per_ei = args.batch_per_ei
    num_rollouts = args.num_rollouts
    wandb_run_name = args.wandb_run_name
    vllm_sampling_params["n"] = num_rollouts
    output_dir = f"/home/RESULTS/{wandb_run_name}"
    num_epochs = args.num_epochs
    model_name = args.model_name
    
    input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
    
    # -------------------------------------------------------------#
    # Initialize wandb
    # -------------------------------------------------------------#
    pretty_print(f"Initializing wandb project {wandb_project} and run {wandb_run_name}...", title="Wandb initialization", is_super_title=True)
    wandb.init(project=wandb_project, name=wandb_run_name, config=input_config)

    # x/y-axis for train and eval steps
    wandb.define_metric("step_train")
    wandb.define_metric("step_eval")

    # metrics that start with train/ are tied to train_step and vice versa for eval/
    wandb.define_metric("train/*", step_metric="step_train")
    wandb.define_metric("eval_ei/*", step_metric="step_eval")
    pretty_print(input_config, title="Input config", is_super_title=True)
    
    
    # -------------------------------------------------------------#
    # Assertions and other setup
    # -------------------------------------------------------------#
    # assertions to ensure the training can be run
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please use a GPU for training")
    if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype = "float16"
        vllm_dtype = "float16"
        raise ValueError("bfloat16 is not supported on this device, please use a different dtype")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    # -------------------------------------------------------------#
    # Initialize the vLLM model for inference 
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the vLLM model for inference...", title="vLLM model initialization", is_super_title=True)
    vllm_init_params = {
        "model": model_name,
        "gpu_memory_utilization": vllm_gpu_memory_utilization,
        "max_model_len": vllm_max_model_len,
        "max_num_seqs": vllm_max_num_seqs
    }
    pretty_print(vllm_init_params, title="vLLM model initialization parameters")
    vllm_model = init_vllm(seed, vllm_init_params)
    # create sampling parameters object
    vllm_sampling_params_obj = SamplingParams(**vllm_sampling_params)
    vllm_sampling_params_obj_eval = SamplingParams(**vllm_sampling_params_eval)
    
    
    # -------------------------------------------------------------#
    # Initialize the tokenizer and model
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the tokenizer and model...", title="Tokenizer and model initialization", is_super_title=True)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attention_type, device_map=device)
    # compile the model
    if use_compile:
        print(f"Compiling the model...")
        model = torch.compile(model)
    
    
    # -------------------------------------------------------------#
    # Setup the optimizer
    # -------------------------------------------------------------#
    pretty_print(f"Initializing the optimizer...", title="Optimizer initialization", is_super_title=True)
    # check if fused AdamW is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    print(f"Using fused AdamW: {use_fused}")
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_list[0], fused=use_fused)
    print(optimizer)
    
    
    # -------------------------------------------------------------#
    # Prepare the validation data
    # -------------------------------------------------------------#
    if run_intermediate_eval:
        pretty_print(f"Preparing the validation data of {max_val_examples} examples from {val_data_file}...", title="Validation data preparation", is_super_title=True)
        val_prompts, val_baseline_results = prepare_val_data(val_data_file, prompt_template_file, max_val_examples)
        # print the validation prompts and baseline results
        pretty_print(val_prompts[:2], title="Example validation prompt")
        pretty_print(val_baseline_results[2], title="Example validation baseline result")
    
    
    # -------------------------------------------------------------#
    # Expert iteration loop
    # -------------------------------------------------------------#
    model.train()
    
    # clean all the temporary files
    for f in os.listdir(tmp_dir):
        if f.startswith('tmp_') and f.endswith('.json'):
            os.remove(os.path.join(tmp_dir, f))

    # outer loop: expert iteration
    ei_step = 0
    train_step_counter = 0
    
    # # evaluate the model on the validation data
    # pretty_print(f"Evaluating the model on the validation data...", title="Validation data evaluation")
    # t_eval = time.time()
    # vllm_eval_results = evaluate_vllm(vllm_model, r1_zero_reward_fn, val_prompts, val_baseline_results, vllm_sampling_params_obj_eval)
    # dt = time.time() - t_eval
    # # print the evaluation metrics
    # print(f"exp_iter: {ei_step} | avg_acc: {vllm_eval_results['accuracy']['avg_acc']:.4f} | avg_format_acc: {vllm_eval_results['accuracy']['avg_format_acc']:.4f} | dt: {dt:.2f}s")
    # # log the evaluation metrics to wandb
    # eval_log_dict = {
    #             "step_eval": ei_step,
    #             "eval_ei/avg_acc": vllm_eval_results['accuracy']['avg_acc'],
    #             "eval_ei/avg_format_acc": vllm_eval_results['accuracy']['avg_format_acc'],
    #             "eval_ei/dt": dt
    #         }
    # wandb.log(eval_log_dict)
    # torch.cuda.empty_cache()

    for ei_step in range(1, num_ei + 1):
        pretty_print(f"Starting expert iteration {ei_step}...", title=f"Expert iteration {ei_step}", is_super_title=True)

        # sample a batch of batch_per_ei examples from the data and filter them and save the filtered data to a jsonl file
        pretty_print(f"Sampling and filtering a batch of {batch_per_ei} examples from the data...", title="Filtering train data")
        tmp_train_data_file = f"{tmp_dir}/tmp_{ei_step}.json"
        _, num_filtered_examples = create_ei_filtered_data(prompt_template_file, train_data_file, batch_per_ei, vllm_model, vllm_sampling_params_obj, r1_zero_reward_fn, tmp_train_data_file, use_generated_reasoning)
        print(f"Filtered data saved to {tmp_train_data_file}")
        
        # Setup the data loader
        pretty_print(f"Initializing the data loader...", title="Data loader initialization")
        train_data_loader = SftTrainDataLoaderLite(tmp_train_data_file, prompt_template_file, micro_batch_size, seed + ei_step)
        
        # get lr, max_steps, and cur_grad_acc_steps
        learning_rate, max_steps, cur_grad_acc_steps = get_lr_and_max_steps(num_filtered_examples, batch_boundary_list, total_batch_size_list, learning_rate_list, train_data_loader, micro_batch_size, num_epochs)
        # update the learning rate in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            
        pretty_print(f"Training the model for {max_steps} steps...", title="Training loop")
        for step in range(max_steps):
            # start time    
            t0 = time.time()
            
            # set the optimizer to zero grad
            optimizer.zero_grad(set_to_none=True)
            
            loss_accum = 0.0
            entropy_accum = 0.0
            total_response_tokens = 0
            for micro_step in range(cur_grad_acc_steps):
                # get a batch of data
                batch = train_data_loader.get_batch()
                # tokenize the batch
                tokenized_batch = tokenize_prompt_and_output(prompt_strs=[item["prompt"] for item in batch],
                                                            output_strs=[item["response"] for item in batch],
                                                            tokenizer=tokenizer)
                input_ids, labels, response_mask = tokenized_batch["input_ids"].to(device), tokenized_batch["labels"].to(device), tokenized_batch["response_mask"].to(device)
                
                # calculate the loss
                with torch.autocast(device_type=device_type, dtype=dtype):
                    # get the log-probs of the response given the prompt and the token entropy
                    # log_probs: (batch_size, sequence_length), token_entropy: (batch_size, sequence_length)
                    response_log_probs = get_response_log_probs(model, input_ids, labels, True)
                    # calculate the loss
                    loss, loss_metrics = sft_microbatch_train_step(response_log_probs["log_probs"], response_mask, cur_grad_acc_steps, 
                                                                normalize_constant=normalize_constant, per_token_loss=use_per_token_loss)
                    # accumulate the loss
                    loss_accum += loss.detach()
                
                # accumulate the token entropy (only for response tokens)
                # move it outside the autocast to save memory overhead (minor)
                entropy_accum += (response_log_probs["token_entropy"] * response_mask).sum().detach()
                total_response_tokens += response_mask.sum().detach()
                
            # global norm gradient clipping at 1.0
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            # update the model parameters
            optimizer.step()
        
        
            # calculate average token entropy across all response tokens
            avg_token_entropy = (entropy_accum / total_response_tokens).item()
            # print the loss metrics
            dt = time.time() - t0
            print(f"step: {train_step_counter:04d} | loss: {loss_accum.item():.4f} | avg_entropy: {avg_token_entropy:.4f} | lr: {learning_rate:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}s")
            train_step_counter += 1
            
            train_log_dict = {
                "step_train": train_step_counter,
                "train/loss": loss_accum.item(),
                "train/avg_token_entropy": avg_token_entropy,
                "train/lr": learning_rate,
                "train/norm": norm.item(),
                "train/dt": dt
            }
            wandb.log(train_log_dict)

        # load the model weights to the vLLM model
        pretty_print(f"Loading the model weights from the previous expert iteration to vLLM model...", title="vLLM model loading")
        load_policy_into_vllm_instance(model, vllm_model)
        
        # run intermediate evaluation on the validation data
        if run_intermediate_eval and ei_step in eval_ei_steps:
            torch.cuda.empty_cache()
            pretty_print(f"Evaluating the model on the validation data...", title="Validation data evaluation")
            
            # evaluate the model on the validation data
            t_eval = time.time()
            vllm_eval_results = evaluate_vllm(vllm_model, r1_zero_reward_fn, val_prompts, val_baseline_results, vllm_sampling_params_obj_eval)
            dt = time.time() - t_eval

            # print the evaluation metrics
            print(f"exp_iter: {ei_step} | avg_acc: {vllm_eval_results['accuracy']['avg_acc']:.4f} | avg_format_acc: {vllm_eval_results['accuracy']['avg_format_acc']:.4f} | dt: {dt:.2f}s")
            
            eval_log_dict = {
                "step_eval": ei_step,
                "eval_ei/avg_acc": vllm_eval_results['accuracy']['avg_acc'],
                "eval_ei/avg_format_acc": vllm_eval_results['accuracy']['avg_format_acc'],
                "eval_ei/dt": dt
            }
            wandb.log(eval_log_dict)
            torch.cuda.empty_cache()
            if ei_step in save_ei_steps:
                ckpt_dir = output_dir + f"/checkpoint_ei_{ei_step}"
                print(f"Saving the model weights to {ckpt_dir}...")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
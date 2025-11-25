import json
import random
from typing import Callable, List

from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


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

# -------------------------------------------------------------#
# Evaluate a LLM on a list of prompts, compute evaluation metrics, and return the results
# -------------------------------------------------------------#
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    baseline_results: List[dict],
    sampling_params: SamplingParams
) -> None:
    """
    Evaluate a LLM on a list of prompts, compute evaluation metrics, and return the results
    """
    
    # generate the prompts
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # calculate the reward using the reward function
    # and add the outputs to the baseline results
    acc_dict = {"avg_acc": 0.0,
                "avg_format_acc": 0.0,
                }
    for i, (output, baseline_result) in enumerate(zip(outputs, baseline_results)):
        output_text = output.outputs[0].text
        reward = reward_fn(output_text, str(baseline_result["expected_answer"]).strip())
        baseline_result["reward"] = reward
        baseline_result["output"] = output_text
        acc_dict["avg_acc"] += reward["reward"]
        acc_dict["avg_format_acc"] += reward["format_reward"]
       
    total_examples = len(baseline_results)
    for key in acc_dict.keys():
        acc_dict[key] /= total_examples
    
    # save the baseline results to a jsonl file
    baseline_results_with_eval = {
        "results": baseline_results,
        "accuracy": acc_dict
    }

    return baseline_results_with_eval

# -------------------------------------------------------------#
# functions to initialize the vLLM model and load the policy into the vLLM model during training
# -------------------------------------------------------------#
def init_vllm(seed: int, vllm_init_params: dict):
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
    return LLM(**vllm_init_params)

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    # If the model was torch.compiled, the real module is in ._orig_mod
    if hasattr(policy, "_orig_mod"):
        policy_for_state = policy._orig_mod
    else:
        policy_for_state = policy
    state_dict = policy_for_state.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# -------------------------------------------------------------#
# Prepare the validation data
# -------------------------------------------------------------#
def prepare_val_data(val_data_file: str, prompt_template_file: str, max_val_examples: int=100):
    """
    Prepare the validation data.
    """
    # read the val data file and prompt template file
    with open(prompt_template_file, "r") as f:
        prompt_template = f.read()
    with open(val_data_file, "r") as f:
        val_data = json.load(f)

    # sample the validation data
    total_val_examples = len(val_data)
    if max_val_examples > total_val_examples:
        print(f"Warning: max_val_examples={max_val_examples} is greater than the total_val_examples={total_val_examples}, using all {total_val_examples} examples")
    else:
        val_data = random.sample(val_data, max_val_examples)
    
    # create the list of prompts and baseline results dict
    prompts, baseline_results = zip(*[
        (prompt_template.format(question=val_data[i]["problem"]), 
        {
            "problem": val_data[i]["problem"], 
            "expected_answer": val_data[i]["expected_answer"]
        }
        )
        for i in range(len(val_data))
    ])
        
    return list(prompts), baseline_results

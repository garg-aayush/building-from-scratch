import json
from typing import Callable, List

from vllm import LLM, SamplingParams


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
        acc_dict["avg_format_acc"] += reward["format_reward"] if reward["format_reward"] == 1.0 and reward["answer_reward"] == 0.0 else 0.0
       
    total_examples = len(baseline_results)
    for key in acc_dict.keys():
        acc_dict[key] /= total_examples
    
    # save the baseline results to a jsonl file
    baseline_results_with_eval = {
        "results": baseline_results,
        "accuracy": acc_dict
    }
    with open(eval_file, "w") as f:
        f.write(json.dumps(baseline_results_with_eval, indent=2))

    return baseline_results_with_eval
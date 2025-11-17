import json

from utils.drgrpo_grader import r1_zero_reward_fn
from utils.helper_fns import evaluate_vllm, pretty_print
from vllm import LLM, SamplingParams

# -------------------------------------------------------------#
# Input params
# -------------------------------------------------------------#
val_file = "data/val.jsonl"
eval_file = "data/baseline_results.jsonl"
prompt_template_file = "data/r1_zero.prompt"

# model name
model_name = "Qwen/Qwen2.5-Math-1.5B"

# sampling parameters
sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True
}

# model parameters
model_params = {
    "model": model_name,
    "max_model_len": 2048,
    "max_num_seqs": 96
}

# print config keys
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")


# -------------------------------------------------------------#
# Load the validation data
# -------------------------------------------------------------#
with open(val_file, "r") as f:
    val_data = json.load(f)
print(f"Loaded {len(val_data)} val examples from {val_file}")
pretty_print(val_data[0], title="val. example")

# -------------------------------------------------------------#
# Load the prompt template
# -------------------------------------------------------------#
with open(prompt_template_file, "r") as f:
    prompt_template = f.read()
pretty_print(prompt_template, title="Prompt template")


# -------------------------------------------------------------#
# Create the list of prompts and baseline results dict
# -------------------------------------------------------------#
prompts, baseline_results = zip(*[
    (prompt_template.format(question=val_data[i]["problem"]), 
    {
        "problem": val_data[i]["problem"], 
        "expected_answer": val_data[i]["expected_answer"]
    }
    )
    for i in range(len(val_data))
])
pretty_print(prompts[0], title="Example prompt")


# -------------------------------------------------------------#
# Initialize the LLM
# -------------------------------------------------------------#
pretty_print(f"Initializing the {model_name} model...", title="LLM initialization")
# sampling parameters object
sampling_params = SamplingParams(**sampling_params)
# create LLM object
llm = LLM(**model_params)


# -------------------------------------------------------------#
# evaluate
# -------------------------------------------------------------#
# evaluate the model
eval_results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, baseline_results, sampling_params)
pretty_print(eval_results["results"][0], title="Example baseline result")
pretty_print(eval_results["accuracy"], title="Evaluation accuracy")

# save the evaluation results to a jsonl file
print(f"Saving evaluation results to {eval_file}...")
with open(eval_file, "w") as f:
    f.write(json.dumps(eval_results, indent=2))
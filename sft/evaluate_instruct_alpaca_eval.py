import json
import os

# set this in case of any multiprocessing errors
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# make sure to set (this is required for the alpaca_eval script to work with Fireworks API)
os.environ["OPENAI_API_KEY"] = os.getenv("FIREWORKS_API_KEY")
import re
import time

import torch
from utils.helper_fns import pretty_print
from vllm import LLM, SamplingParams

# -------------------------------------------------------------#
# Input params (MMLU) evaluation
# -------------------------------------------------------------#
val_file = "data/eval_data/alpaca_eval/alpaca_eval.jsonl"
system_prompt_file = "data/zero_shot_system_prompt.prompt"
save_only_accuracy = False
model_name = "meta-llama/Llama-3.1-8B"
results_dir = "results/dialogue/baseline"
eval_filename = "baseline_alpaca_eval_outputs.json"
generator_name = "llama-3.1-8b-base"
seed = 1337
reference_outputs_file = "data/eval_data/alpaca_eval/alpaca_eval_gpt4_baseline.json"
annotators_config_dir = "data/eval_data/alpaca_eval/configs"

# sampling parameters
sampling_params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "stop": ["```"]#, "# Query:"],
}

# model parameters
model_params = {
    "model": model_name,
    "max_model_len": 2048, # purely for mem. considerations (input + output)
    "max_num_seqs": 32 # purely for mem. considerations
}

# create the eval_file directory if it doesn't exist
eval_file = os.path.join(results_dir, eval_filename)
os.makedirs(os.path.dirname(eval_file), exist_ok=True)

# print config keys
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")


# -------------------------------------------------------------#
# Load the prompt templates
# -------------------------------------------------------------#
with open(system_prompt_file, "r") as f:
    prompt_template = f.read()
pretty_print(prompt_template, title="prompt template")


# -------------------------------------------------------------#
# Initialize the LLM
# -------------------------------------------------------------#
pretty_print(f"Initializing the {model_name} model...", title="LLM initialization")
# sampling parameters object
sampling_params = SamplingParams(**sampling_params)
# create LLM object
llm = LLM(**model_params)

# -------------------------------------------------------------#
# Evaluate the model
# -------------------------------------------------------------#
# get all the csv files in the val_dir
results = {}
acc_dict = {}

# read the jsonl file
with open(val_file, "r") as f:
    val_data = [json.loads(line) for line in f]
pretty_print(val_data[0], title="Example val data")

def create_prompt(prompt_template: str, val_example: dict[str, str]):
    """Create a prompt for a given subject and question dictionary (MMLU format)."""
    prompt = prompt_template.format(instruction=val_example["instruction"],
                                    )
    return prompt

def get_prompts(prompt_template: str, val_data: list[dict[str, str]]):
    """Get a list of prompts and answers from a jsonl file."""
    prompts, datasets = zip(*[(create_prompt(prompt_template, val_example), val_example["dataset"]) for val_example in val_data])
    return list(prompts), list(datasets)

prompts, datasets = get_prompts(prompt_template, val_data)

# generate the results
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print(f"Generated {len(outputs)} outputs in {time.time() - t0:.2f}s")

results = []

for prompt, output, dataset, val_example in zip(prompts, outputs, datasets, val_data):
    results.append({
        "prompt": prompt,
        "instruction": val_example["instruction"],
        "output": output.outputs[0].text,
        "generator": generator_name,
        "dataset": dataset,
    })

print(f"Saving results to {eval_file}...")
with open(eval_file, "w") as f:
    json.dump(results, f, indent=2)
    
    
# run the alpaca_eval script
print("Running alpaca_eval script...")
os.system(f"alpaca_eval --model_outputs {eval_file} --reference_outputs {reference_outputs_file} --annotators_config {annotators_config_dir} --seed {seed} --base-dir '.'")
# rename the annotations file 
print("Renaming annotations and leaderboard files...")
os.system(f"mv -v {results_dir}/annotations.json {results_dir}/baseline_alpaca_eval_annotations.json")
# rename the leaderboard file
os.system(f"mv -v {results_dir}/leaderboard.csv {results_dir}/baseline_alpaca_eval_leaderboard.csv")
# delete 
print("Deleting intermediate files...")
os.system(f"rm -v {annotators_config_dir}/annotations_seed{seed}_config.json")


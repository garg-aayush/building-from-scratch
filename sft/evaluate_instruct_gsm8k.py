import json
import os

# set this in case of any multiprocessing errors
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
import re
import time

import torch
from utils.helper_fns import pretty_print
from vllm import LLM, SamplingParams

# -------------------------------------------------------------#
# Input params (MMLU) evaluation
# -------------------------------------------------------------#
val_file = "data/eval_data/gsm8k/test.jsonl"
system_prompt_file = "data/zero_shot_system_prompt.prompt"
gsm8k_question_file = "data/question_only.prompt"
save_only_accuracy = False
model_name = "meta-llama/Llama-3.1-8B"
results_dir = "results/dialogue/baseline"
eval_filename = "baseline_gsm8k_results.jsonl"
headers = ["question", "answer"]
answer_col = headers[-1]

# sampling parameters
sampling_params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
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
accuracy_file = eval_file.replace("_results.jsonl", "_accuracy.jsonl")
os.makedirs(os.path.dirname(eval_file), exist_ok=True)

# print config keys
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")


# -------------------------------------------------------------#
# Load the prompt templates
# -------------------------------------------------------------#
with open(system_prompt_file, "r") as f:
    system_prompt = f.read()
pretty_print(system_prompt, title="System prompt")
with open(gsm8k_question_file, "r") as f:
    gsm8k_question_prompt = f.read()
pretty_print(gsm8k_question_prompt, title="GSM8K question prompt")
prompt_template = system_prompt.format(instruction=gsm8k_question_prompt)
# prompt_template = """
# {question}
# Answer:
# """
pretty_print(prompt_template, title="Prompt template")


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

def parse_gsm8k_response(response: str) -> str:
    """Parse the GSM8K response to get the answer."""
    if not isinstance(response, str):
        return None
    # parse the response to get the extracted answer
    # r'\d+' matches one or more digits, find all matches in the response
    num_re = re.compile(r'\d+')
    all_matches = num_re.findall(response)
    # return the last match if found
    return all_matches[-1] if all_matches else None

def create_prompt(prompt_template: str, val_example: dict[str, str]):
    """Create a prompt for a given subject and question dictionary (MMLU format)."""
    prompt = prompt_template.format(question=val_example["question"],
                                    )
    return prompt

def get_prompts_and_answers(prompt_template: str, val_data: list[dict[str, str]]):
    """Get a list of prompts and answers from a jsonl file."""
    prompts = [create_prompt(prompt_template, val_example) for val_example in val_data]
    answers = [val_example["answer"] for val_example in val_data]
    ground_truths = [parse_gsm8k_response(answer) for answer in answers]
    return list(prompts), list(answers), list(ground_truths)

prompts, answers, ground_truths = get_prompts_and_answers(prompt_template, val_data)

# generate the results
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print(f"Generated {len(outputs)} outputs in {time.time() - t0:.2f}s")

results = []
acc_dict = {"accuracy": 0.0}

for prompt, output, answer, ground_truth in zip(prompts, outputs, answers, ground_truths):
    results.append({
        "prompt": prompt,
        "output": output.outputs[0].text,
        "answer": answer,
        "ground_truth": ground_truth,
        "extracted_answer": parse_gsm8k_response(output.outputs[0].text),
    })
    
    acc_dict["accuracy"] += parse_gsm8k_response(output.outputs[0].text) == ground_truth

acc_dict["accuracy"] /= len(prompts)
pretty_print(acc_dict, title="Accuracy")

print(f"Saving results to {eval_file}...")
with open(eval_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saving accuracy to {accuracy_file}...")
with open(accuracy_file, "w") as f:
    json.dump(acc_dict, f, indent=2)
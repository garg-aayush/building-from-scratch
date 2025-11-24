import json
import os

# set this in case of any multiprocessing errors
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
import re
import time
from typing import Dict, List

import pandas as pd
import torch
from utils.helper_fns import pretty_print
from vllm import LLM, SamplingParams

# -------------------------------------------------------------#
# Input params (MMLU) evaluation
# -------------------------------------------------------------#
val_dir = "data/eval_data/mmlu"
system_prompt_file = "data/zero_shot_system_prompt.prompt"
mmlu_question_file = "data/mmlu_question.prompt"
save_only_accuracy = False
model_name = "meta-llama/Llama-3.1-8B"
results_dir = "results/dialogue/baseline"
eval_filename = "baseline_mmlu_results.jsonl"
headers = ["question", "A", "B", "C", "D", "answer"]
answer_col = headers[-1]

# sampling parameters
sampling_params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 128,
    "stop": ["```"],#, "# Query:"],
}

# model parameters
model_params = {
    "model": model_name,
    "max_model_len": 2048, # purely for mem. considerations (input + output)
    "max_num_seqs": 64 # purely for mem. considerations
}

# create the eval_file directory if it doesn't exist
eval_file = os.path.join(results_dir, eval_filename)
accuracy_file = eval_file.replace("_results.jsonl", "_accuracy.jsonl")
os.makedirs(os.path.dirname(eval_file), exist_ok=True)

# print config keys
input_config = {k: v for k, v in globals().items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, dict))}
pretty_print(input_config, title="Input config")


# -------------------------------------------------------------#
# Helper functions
# -------------------------------------------------------------#
def create_prompt(prompt_template: str, subject: str, q_dict: dict[str, str]):
    """Create a prompt for a given subject and question dictionary (MMLU format)."""
    prompt = prompt_template.format(subject=subject,
                                    question=q_dict["question"],
                                    options=[q_dict["A"], q_dict["B"], q_dict["C"], q_dict["D"]],
                                    )
    return prompt

def get_prompts_and_answers(prompt_template: str, val_dir: str, csv_file: str, headers: List[str], answer_col: str):
    """Get a list of prompts and answers from a csv file."""
    # read the csv file
    df = pd.read_csv(os.path.join(val_dir, csv_file),
                     names=headers)
    # orient="records" to get a list of dictionaries
    qs_dict = df.to_dict(orient="records")
    subject = csv_file.replace("_test.csv", "").replace("_", " ")
    prompts, answers = zip(*[
        (create_prompt(prompt_template, subject, q_dict), q_dict[answer_col]) 
        for q_dict in qs_dict])
    return subject, list(prompts), list(answers)

def append_results(results: dict, acc_dict: dict, subject: str, prompts: List[str], outputs: List, answers: List[str]):
    """Append the results to the results dict."""
    results[subject] = []
    acc_dict[subject] = {"parsed_accuracy": 0.0, "accuracy": 0.0}
    num_examples = len(prompts)
    accuracy = 0.0
    parsed_accuracy = 0.0
    for prompt, output, answer in zip(prompts, outputs, answers):
        generated_text = output.outputs[0].text
        # parse the generated text to get the extracted answer
        extracted_answer, is_correctly_parsed = parse_mmlu_response(generated_text)
        accuracy += str(extracted_answer) == str(answer)
        parsed_accuracy += is_correctly_parsed
        results[subject].append({
            "prompt": prompt,
            "output": generated_text,
            "ground_truth": answer,
            "extracted_answer": extracted_answer,
            "is_correctly_parsed": is_correctly_parsed,
        })
    acc_dict[subject]["parsed_accuracy"] = (parsed_accuracy / num_examples)
    acc_dict[subject]["accuracy"] = (accuracy / num_examples)
    acc_dict[subject]["num_examples"] = num_examples
    return results, acc_dict

def parse_mmlu_response(response: str) -> tuple[str, bool]:
    """Parse the response and return the correct answer and a boolean indicating if the response is correctly parsed"""
    # check if the response is correct
    if not isinstance(response, str):
        return None, False
    # strip of all escape sequences and spaces at start and end
    response = response.upper().strip()
    # patterns to match the extracted answer
    patterns = [
        # Matches exactly one character A/B/C/D with nothing before or after (^ = start, $ = end)
        r"^([ABCD])$",
        r"\b([ABCD])\b",
        r"ANSWER[:： ]*([ABCD])",    # "Answer: C"
        r"OPTION[:： ]*([ABCD])",    # "Option D"
        r"CHOICE[:： ]*([ABCD])",    # "Choice B"
        r"\(([ABCD])\)",             # "(A)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1), True
    return None, False


def weighted_average(in_dict: Dict[str, dict]) -> dict:
    """Calculate the weighted average across multiple subjects"""
    total_count = 0
    weighted_acc = 0.0
    weighted_parsed_acc = 0.0
    for subject, value in in_dict.items():
        weighted_acc += float(value["accuracy"]) * float(value["num_examples"])
        weighted_parsed_acc += float(value["parsed_accuracy"]) * float(value["num_examples"])
        total_count += value["num_examples"]
    out_dict = {
        "accuracy": (weighted_acc / total_count),
        "parsed_accuracy": (weighted_parsed_acc / total_count),
        "num_examples": total_count,
    }
    return out_dict

# -------------------------------------------------------------#
# Load the prompt templates
# -------------------------------------------------------------#
with open(system_prompt_file, "r") as f:
    system_prompt = f.read()
pretty_print(system_prompt, title="System prompt")
with open(mmlu_question_file, "r") as f:
    mmlu_question_prompt = f.read()
pretty_print(mmlu_question_prompt, title="MMLU question prompt")
prompt_template = system_prompt.format(instruction=mmlu_question_prompt)
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
def calculate_accuracy(results: dict) -> dict:
    """Calculate the accuracy of the model"""
    accuracy = {}
    for subject, results in results.items():
        accuracy[subject]["parsed_accuracy"] = sum(result["is_correctly_parsed"] for result in results) / len(results)
        accuracy[subject]["accuracy"] = sum(result["extracted_answer"] == result["ground_truth"] for result in results) / len(results)
    return accuracy

# get all the csv files in the val_dir
results = {}
acc_dict = {}
csv_files = [f for f in os.listdir(val_dir) if f.endswith("_test.csv")]
num_csv_files = len(csv_files)
for i, csv_file in enumerate(csv_files):
    print(f"Loading {csv_file}...")
    prompts = []
    answers = []
    
    # get a list of prompts and answer
    subject, prompts, answers = get_prompts_and_answers(prompt_template, val_dir, csv_file, headers, answer_col)
    
    # generate the results
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"Generated {len(outputs)} outputs for '{subject}' in {time.time() - t0:.2f}s")
    
    # clear
    torch.cuda.empty_cache()
    
    # add the results to the results dict
    # results: dict[str, list[dict[str, str]]]
    # acc_dict: dict[str, dict[str, float]]
    results, acc_dict = append_results(results, acc_dict, subject, prompts, outputs, answers)
    
    pretty_print(acc_dict[subject], title=f"subject ({i+1}/{num_csv_files}) -> {subject}")
    
# get average accuracy across all subjects
acc_dict["all_subjects"] = weighted_average(acc_dict)
pretty_print(acc_dict["all_subjects"], title="All subjects")

# save the results to a jsonl file
print(f"Saving results to {eval_file}...")
with open(eval_file, "w") as f:
    f.write(json.dumps(results, indent=2))

# save the accuracy dictionary to a json file
print(f"Saving accuracy dictionary to {accuracy_file}...")
with open(accuracy_file, "w") as f:
    f.write(json.dumps(acc_dict, indent=2))
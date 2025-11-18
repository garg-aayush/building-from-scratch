"""
Extract answers from batch inference outputs.

This script reads the batch inference JSONL file and extracts answers
from the model responses, typically found in \boxed{} tags.
"""

import json
import re
from pathlib import Path

# Configuration
BATCH_OUTPUT_FILE = '../data/batch-infer-math-train-outputs.jsonl'
BATCH_INPUT_FILE = '../data/train_data_4_batchinfer.jsonl'
TRAIN_FILE = '../data/train.jsonl'
OUTPUT_FILE = '../data/sft.jsonl'

# Prompt prefix to remove from problem
PROMPT_PREFIX = (
    "Please give a short reason and answer the below question. "
    "Make sure put your final answer within \\boxed{}. \n "
)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def get_reasoning_trace(response_text):
    # reasoning trace is the text before the \boxed{}
    reasoning_trace = re.sub(r'\\boxed{(.*)}', '', response_text)
    # remove the "\n\n\\[\n\n\\]" from the reasoning trace end if it exists
    reasoning_trace = re.sub(r'\n\n\\\[\n\n\\\]', '', reasoning_trace)
    # remove the "\n\\[\n\n\\]" from the reasoning trace start if it exists
    reasoning_trace = re.sub(r'\n\\\[\n\n\\\]', '', reasoning_trace)
    return reasoning_trace

def main():
    # Load problems from batch input file
    input_data = load_data(BATCH_INPUT_FILE)
    print(f"Loaded {len(input_data)} problems")
    
    # load batch inference outputs
    batch_outputs = load_data(BATCH_OUTPUT_FILE)
    print(f"Loaded {len(batch_outputs)} batch inference outputs")
    
    # load train data
    with open(TRAIN_FILE, 'r') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} train data")
    
    # only keep batch_outputs where finish_reason is "stop"
    batch_outputs = [output for output in batch_outputs if output['response']['choices'][0]['finish_reason'] == "stop"]
    print(f"Found {len(batch_outputs)} batch inference outputs where finish_reason is 'stop'")
    
    output_data = []
    for output in batch_outputs:
        custom_id = output['custom_id']
        response_text = output['response']['choices'][0]['message']['content']
        
        # extracted answer is the text between \boxed{}
        extracted_answer = re.search(r'\\boxed{(.*)}', response_text)
        
        # get the reasoning trace from the response text
        reasoning_trace = get_reasoning_trace(response_text)

        # now get the problem text as a separate field
        problem = [input_entry for input_entry in input_data if input_entry['custom_id'] == custom_id][0]["body"]["messages"][0]["content"].replace(PROMPT_PREFIX, '')
        
        # now get the expected answer from the train data
        expected_answer = [train_entry for train_entry in train_data if train_entry['problem'] == problem][0]['expected_answer']
        
        # if extracted answer is not None, add it to the output data
        if extracted_answer:
            output_data.append({
                'problem': problem,
                'reasoning_trace': reasoning_trace,
                'extracted_answer': extracted_answer.group(1),
                'expected_answer': expected_answer,
            })
    
    # find number of examples where extracted_answer is correct
    correct_count = 0
    for item in output_data:
        if item['extracted_answer'] == item['expected_answer']:
            correct_count += 1
    print(f"Found {correct_count} examples where extracted_answer is correct")
    print(f"Accuracy: {correct_count / len(output_data):.2%}")
    
    # save output data to jsonl file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(json.dumps(output_data, indent=2))
    print(f"Saved {len(output_data)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
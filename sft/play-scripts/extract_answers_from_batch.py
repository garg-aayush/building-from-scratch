"""
Extract answers from batch inference outputs.

This script reads the batch inference JSONL file and extracts answers
from the model responses, typically found in \boxed{} tags.
"""

import json
import re
from pathlib import Path

# Configuration
BATCH_OUTPUT_FILE = '../data/batch-infer-math-train-outputs_gpt-oss-120b.jsonl'
BATCH_INPUT_FILE = '../data/train_data_4_batchinference_gpt-oss-120b.jsonl'
TRAIN_FILE = '../data/train.jsonl'
OUTPUT_FILE = '../data/sft_gpt-oss-120b.jsonl'
OUTPUT_FILE_FILTERED = '../data/sft_gpt-oss-120b_filtered.jsonl'

# Prompt prefix for math problems
PROMPT_PREFIX = ( "Please first give a short reason and then answer the below question. Make sure to put your reasoning within <think></think> and the final answer within <answer></answer>.\n\n"
)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def get_reasoning_trace(response_text):
    # reasoning trace is the text before the <answer>
    reasoning_trace = re.sub(r'<think>', '', response_text)
    # update the reasoning trace to replace the </think>\n\n<answer> and </think>\n<answer> with </think> <answer> (helps in grading)
    reasoning_trace = reasoning_trace.replace('</think>\n\n<answer>', '</think> <answer>')
    reasoning_trace = reasoning_trace.replace('</think>\n<answer>', '</think> <answer>')
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
        
        # extracted answer is the text between <answer>
        extracted_answer = re.search(r'<answer>(.*)</answer>', response_text)
        
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
    
    # filter out examples where extracted_answer is not correct
    output_data = [item for item in output_data if item['extracted_answer'] == item['expected_answer']]
    with open(OUTPUT_FILE_FILTERED, 'w') as f:
        f.write(json.dumps(output_data, indent=2))
    print(f"Saved {len(output_data)} items to {OUTPUT_FILE_FILTERED}")

if __name__ == "__main__":
    main()
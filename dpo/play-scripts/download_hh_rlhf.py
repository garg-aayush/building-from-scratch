#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
# ]
# ///
"""
Script to download all 4 folders from Anthropic/hh-rlhf dataset.
"""

import gzip
import json
import os
import shutil

from huggingface_hub import hf_hub_download

# Define the dataset repository
REPO_ID = "Anthropic/hh-rlhf"
REPO_TYPE = "dataset"
SUBFOLDERS = [
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled"
]
FILES = ["train.jsonl.gz", "test.jsonl.gz"]

# Output directory
OUTPUT_DIR = "../data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_all_files(unzip=True):
    """Download all .jsonl.gz files from all subfolders."""
    print(f"Downloading files from {REPO_ID}")
    
    all_jsonl_paths = []
    for subfolder in SUBFOLDERS:
        print(f"Processing subfolder: {subfolder}")
        
        subfolder_path = os.path.join(OUTPUT_DIR, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        
        for file in FILES:
            repo_file_path = f"{subfolder}/{file}"
            gz_path = os.path.join(OUTPUT_DIR, repo_file_path)
            jsonl_path = gz_path.replace(".gz", "")
            all_jsonl_paths.append(jsonl_path)
            if os.path.exists(jsonl_path):
                print(f"  {file} already exists, skipping download")
                continue
            
            print(f"  Downloading {file}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_file_path,
                repo_type=REPO_TYPE,
                local_dir=OUTPUT_DIR
            )
            
            if unzip:
                print(f"  Unzipping {file}...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(jsonl_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  {file} unzipped")
                os.remove(gz_path)
    
    print(f"All files saved to: {OUTPUT_DIR}")
    return all_jsonl_paths

def read_jsonl(file_path):
    """
    Reads a JSONL format file and returns a list of JSON objects.
    """
    json_dicts = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                json_dicts.append(json.loads(line))
    return json_dicts

def create_single_turn_examples(json_dicts, subset_name: str="", data_type: str="train"):
    """
    Creates single-turn examples from multi-turn conversations.
    """
    multi_turn_count = 0
    examples = []
    
    for i,conversation in enumerate(json_dicts):
        # Check both 'chosen' and 'rejected' fields for multi-turn conversations
        for field in ['chosen', 'rejected']:
            text = conversation.get(field, '')
            human_count = text.count('Human:')
            assistant_count = text.count('Assistant:')
            if human_count > 1 or assistant_count > 1:
                multi_turn_count += 1
                break
        if human_count == 1 and assistant_count == 1:
            instruction = conversation['chosen'].split('\n\nAssistant:')[0].split('\n\nHuman:')[1].strip()
            chosen_answer = conversation.get('chosen', '').split('\n\nAssistant:')[1].strip()
            rejected_answer = conversation.get('rejected', '').split('\n\nAssistant:')[1].strip()
            examples.append({
                'instruction': instruction,
                'chosen_answer': chosen_answer,
                'rejected_answer': rejected_answer,
                'subset_name': subset_name,
                'data_type': data_type,
            })
    return multi_turn_count, examples

if __name__ == "__main__":
    # download all the files
    all_jsonl_paths = download_all_files(unzip=True)
    
    # read all the jsonl files and create single turn examples
    print("Create single-turn examples..")
    all_examples = []
    for jsonl_path in all_jsonl_paths:
        print(f"  Processing {jsonl_path}...")
        json_dicts = read_jsonl(jsonl_path)
        subset_name = jsonl_path.split('/')[-2]
        data_type = jsonl_path.split('/')[-1].replace('.jsonl', '')
        multi_turn_count, examples = create_single_turn_examples(json_dicts, subset_name, data_type)
        print(f"  Single-turn examples: {len(examples)}")
        print()
        all_examples.extend(examples)
    
    # save the examples to a jsonl file
    output_file_path = f"{OUTPUT_DIR}/examples.jsonl"
    with open(output_file_path, 'w') as f:
        print(f"Saving {len(all_examples)} examples to {output_file_path}...")
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
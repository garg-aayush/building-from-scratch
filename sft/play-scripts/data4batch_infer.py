"""
Create dataset for Fireworks AI batch inference jobs.

This script reads training data and formats it for batch inference using
the deepseek-v3p1-terminus model on Fireworks AI.
"""

import json
import os
from pathlib import Path

# Configuration
INPUT_TRAIN_FILE = '../data/train.jsonl'
OUTPUT_BATCH_FILE = '../data/train_data_4_batchinference_gpt-oss-120b.jsonl'
MODEL_NAME = 'accounts/fireworks/models/gpt-oss-120b'

# Prompt prefix for math problems
PROMPT_PREFIX = ( "Please first give a short reason and then answer the below question. Make sure to put your reasoning within <think></think> and the final answer within <answer></answer>.\n\n"
)

# Inference parameters for batch job (set them on UI)
INFERENCE_PARAMS = {
    "max_tokens": 1024,
    "top_p": 1,
    "top_k": 40,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "temperature": 0.6
}

def create_batch_inference_payloads(data, limit=None):
    output = []
    data_to_process = data[:limit] if limit else data
    
    for i, item in enumerate(data_to_process):
        messages = [
            {
                "role": "user",
                "content": f"{PROMPT_PREFIX}{item['problem']}"
            }
        ]
        
        payload = {
            "custom_id": f"request-{i}",
            "body": {
            "messages": messages,
            },
            # **INFERENCE_PARAMS
        }
        
        output.append(payload)
    
    return output

def main():
    print("Loading training data...")
    with open(INPUT_TRAIN_FILE, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {INPUT_TRAIN_FILE}")
    
    # Create batch inference payloads
    print("Creating batch inference payloads...")
    payloads = create_batch_inference_payloads(data)
    print(f"Created {len(payloads)} payloads")
    
    # Save to JSONL file
    with open(OUTPUT_BATCH_FILE, 'w') as f:
        for item in payloads:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(payloads)} items to {OUTPUT_BATCH_FILE}")
    
    
if __name__ == "__main__":
    main()

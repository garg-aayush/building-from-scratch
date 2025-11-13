"""
Create train and val datasets (jsonl files) by filtering out validation examples from math12k dataset.
"""

import json

from datasets import concatenate_datasets, load_dataset

# you can download math_results.jsonl from 
# https://github.com/Christine8888/assignment5-alignment/blob/main/cs336_alignment/math_results.jsonl
INPUT_VAL_FILE = '../data/math_results.jsonl'
MATH_DATASET_NAME = 'hiyouga/math12k'
OUTPUT_TRAIN_FILE = '../data/train.jsonl'
OUTPUT_VAL_FILE = '../data/val.jsonl'


def extract_prob_from_prompt(prompt):
    """Extract the problem text from the prompt."""
    return prompt.split("\nUser:")[1].split("\nAssistant:")[0].strip()


def main():
    # Read validation data
    print("Loading validation data...")
    with open(INPUT_VAL_FILE, 'r') as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation records")
    
    # Extract problem field from validation data
    print("Extracting problems from validation data...")
    val_list = [
        {
            'expected_answer': item['expected_answer'],
            'problem': extract_prob_from_prompt(item['prompt'])
        }
        for item in val_data
    ]
    print(f"Found {len(val_list)} unique validation problems")
    
    # Load and combine math12k dataset
    print(f"Loading dataset: {MATH_DATASET_NAME}")
    ds = load_dataset(MATH_DATASET_NAME)
    ds_combined = concatenate_datasets([ds["train"], ds["test"]])
    del ds
    print(f"Combined dataset size: {len(ds_combined)} rows")
    
    # Filter out validation problems to create training set
    print("Filtering out validation problems...")
    train_ds = ds_combined.filter(lambda x: x["problem"] not in {item["problem"] for item in val_list})
    # Rename "answer" column to "expected_answer" for consistency
    train_ds = train_ds.rename_column("answer", "expected_answer")
    print(f"Found {len(train_ds)} training problems")
    
    # Convert to list of dictionaries
    train_list = [dict(item) for item in train_ds]
    
    # Save as JSONL (one JSON object per line)
    print(f"Saving training dataset to {OUTPUT_TRAIN_FILE}...")
    with open(OUTPUT_TRAIN_FILE, 'w') as f:
        f.write(json.dumps(train_list, indent=2))
    print(f"Saving validation dataset to {OUTPUT_VAL_FILE}...")
    with open(OUTPUT_VAL_FILE, 'w') as f:
        f.write(json.dumps(val_list, indent=2))

if __name__ == "__main__":
    main()

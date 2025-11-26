import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils.helper_fns import pretty_print


class InstructFinetuneDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, dataset_path: str, seq_length: int, shuffle: bool = True, alpaca_prompt_template_file: str = "data/alpaca_sft.prompt", apply_masking: bool = False):
        """Initialize the InstructFinetuneDataset.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
            dataset_path (str): The path to the dataset.
            seq_length (int): The sequence length.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            alpaca_prompt_template_file (str, optional): The path to the alpaca_sft.prompt file. Defaults to "data/alpaca_sft.prompt".
            apply_masking (bool, optional): Whether to apply masking to the data (prompts only). If True, the prompt tokens will be masked as -100 for the loss function. Defaults to False.
        """
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.apply_masking = apply_masking
        
        # 1) Load the data
        examples = self._load_data()
        print(f"Read {len(examples)} examples from {self.dataset_path}")
        
        # 2) Read the alpaca_sft.prompt file
        with open(alpaca_prompt_template_file, "r") as f:
            self.ALPACA_SFT_PROMPT_TEMPLATE = f.read()
        
        # 3) Shuffle the data
        if self.shuffle:
            print(f"Shuffling the data...")
            random.shuffle(examples)
            
        # 4) Create full prompt examples
        full_prompts_examples = self._apply_prompt_template(examples, include_response=True)
        eos_token = self.tokenizer.eos_token
        full_prompts_examples = [example + eos_token for example in full_prompts_examples]
        
        # 5) tokenize the full prompts examples
        print(f"Tokenizing the full prompts examples...")
        tokenized_full_prompts_examples = self.tokenizer(full_prompts_examples).input_ids
        
        # 6) tokenize the prompts only examples if apply_masking is True
        if self.apply_masking:
            print(f"Applying prompt template to prompts only since apply_masking is {self.apply_masking}...")
            prompts_only_examples = self._apply_prompt_template(examples, include_response=False)
            tokenized_prompts_only_examples = self.tokenizer(prompts_only_examples).input_ids
        else:
            tokenized_prompts_only_examples = None
        
        # 7) Pack the input ids and labels as tensors
        print(f"Packing the input ids and labels into tensors...")
        self.all_input_ids, self.all_labels = self._pack_input_ids_and_labels(tokenized_full_prompts_examples, tokenized_prompts_only_examples)
        
        
    def _pack_input_ids_and_labels(self, tokenized_examples, tokenized_prompts_only_examples=None):
        all_input_ids = []
        all_labels = []
        
        iterator = zip(tokenized_prompts_only_examples, tokenized_examples) if self.apply_masking else tokenized_examples
        for i, example in enumerate(iterator):
            if self.apply_masking:
                prompt_ids, input_ids = example
                labels = list(input_ids)
                # drop the last token from the prompts_id to avoid token merging at the boundary
                prompt_ids = prompt_ids[:-1]
                
                # Verify the prompt tokens didn't change during full encoding
                if not torch.equal(torch.tensor(input_ids[:len(prompt_ids)]), torch.tensor(prompt_ids)):
                    print(f"WARNING: Tokenization mismatch detected at index {i}!")
                    print("The start of the full sequence does not match the prompt-only sequence. This suggests the prompt template is causing token merging at the boundary.")
                
                # mask the labels as -100 for the prompt tokens
                # -100 as pytorch loss function ignores these tokens, this is an arbitrary choice
                mask_len = min(len(prompt_ids), len(labels))
                labels[:mask_len] = [-100] * mask_len
            else:
                input_ids = example
                # list ensures the example is a separate tensor
                labels = list(example)
            all_input_ids.extend(input_ids)
            all_labels.extend(labels)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        return all_input_ids, all_labels

    def _apply_prompt_template(self, examples, include_response: bool = True):
        formatted_examples = []
        for example in examples:
            response = example["response"] if include_response else ""
            formatted_examples.append(self.ALPACA_SFT_PROMPT_TEMPLATE.format(instruction=example["prompt"], response=response))
        return formatted_examples

    def _load_data(self):
        with open(self.dataset_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        # dropping the last incomplete batch (sort of standard practice in packing datasets)
        return len(self.all_input_ids) // self.seq_length
    
    def __getitem__(self, idx):
        if idx <0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self)}")
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        return {
            "input_ids": self.all_input_ids[start_idx:end_idx],
            "labels": self.all_labels[start_idx:end_idx]
        }
        

def iterate_dataset(dataset: InstructFinetuneDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False):
    """Iterate over the dataset and yield batches of data.

    Args:
        dataset (InstructFinetuneDataset): The dataset to iterate over.
        batch_size (int): The batch size.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): The number of workers to use for the data loader. Defaults to 0.
        pin_memory (bool, optional): Whether to pin the memory of the data. Defaults to False.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

import json

import numpy as np


class SftTrainDataLoaderLite:
    """A lightweight data loader for SFT that loads training data, applies prompt templates, and yields batches.
    Automatically reshuffles and resets when reaching the end of the dataset for continuous epoch training."""
    
    def __init__(self, data_file: str, prompt_template_file: str, batch_size: int, seed: int = 42):
        """Initialize the data loader, load data from file, apply prompt template, and shuffle the dataset."""
        self.data_file = data_file
        self.prompt_template_file = prompt_template_file
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        
        # load the data
        self._load_data()
        # shuffle the data
        self.rng.shuffle(self.data)
        # pointer to the current batch
        self.ptr = 0
        # size of the data
        self.data_size = len(self.data)

    def _load_data(self):
        """Load JSON data from file, apply prompt template to each example, and construct the dataset."""
        # read the data file
        with open(self.data_file, "r") as f:
            input_data = json.load(f)
        print(f"Loaded {len(input_data)} data points from {self.data_file}")
        
        # read the prompt template file
        print(f"Loading prompt template from {self.prompt_template_file}...")
        with open(self.prompt_template_file, "r") as f:
            prompt_template = f.read()
        
        # create the data
        self.data = []
        for item in input_data:
            prompt = prompt_template.format(question=item["problem"])
            self.data.append({
                "prompt": prompt,
                "response": item["reasoning_trace"],
                "ground_truth": item["expected_answer"]
            })
    
    def get_batch(self):
        """Return the next batch of data and automatically reset/reshuffle when reaching the end of the dataset."""
        batch_data = self.data[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size
        
        # reset
        if self.ptr + self.batch_size >= self.data_size:
            self.reset()
        return batch_data
    
    def reset(self, only_ptr: bool = False):
        """Reset the data pointer to the beginning and optionally reshuffle the dataset."""
        if not only_ptr:
            print(f"Shuffling data...")
            self.rng.shuffle(self.data)
        print(f"Resetting pointer to 0")
        self.ptr = 0
    
    def __len__(self):
        """Return the total number of examples in the dataset."""
        return len(self.data)
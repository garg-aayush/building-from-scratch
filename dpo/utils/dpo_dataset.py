"""
Dataset class for DPO finetuning.
"""

import json
import random

from torch.utils.data import Dataset


class DpoFinetuneDataset(Dataset):
    def __init__(self, dataset_path: str, shuffle: bool = True, split: str="train", num_val=500, seed: int = 42) -> None:
        """
        Initialize the dataset for DPO finetuning.
        Args:
            dataset_path: Path to the dataset jsonl file
            shuffle: Whether to shuffle the dataset
            split: Whether to use the train or val split
            num_val: Number of examples to use for validation
            seed: Random seed for consistent shuffling/splitting
        """
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.split = split
        self.num_val = num_val
        self.seed = seed
        

        # read dataset jsonl
        self.dataset = self._load_data()
        print(f"Read {len(self.dataset)} examples from {dataset_path}")
        
        # shuffle the dataset
        if shuffle:
            # Use a local random instance with a fixed seed to ensure 
            # train and val splits are consistent across different initializations
            r = random.Random(seed)
            r.shuffle(self.dataset)
        
        # split the dataset
        if self.split == "train":
            self.dataset = self.dataset[self.num_val:]
            print(f"Using {len(self.dataset)} examples for training")
        elif self.split == "val":
            self.dataset = self.dataset[:self.num_val]
            print(f"Using {len(self.dataset)} examples for validation")
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def _load_data(self) -> list[dict]:
        with open(self.dataset_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self)}")
        
        return {
            'prompt': self.dataset[idx]['instruction'].strip(), 
            'chosen_response': self.dataset[idx]['chosen_answer'].strip(), 
            'rejected_response': self.dataset[idx]['rejected_answer'].strip()
        }


if __name__ == "__main__":

    # Path relative to workspace root
    dataset_path = "data/examples.jsonl"
    
    # Test consistent splitting
    num_val = 2
    train_dataset = DpoFinetuneDataset(dataset_path, split="train", num_val=num_val)
    val_dataset = DpoFinetuneDataset(dataset_path, split="val", num_val=num_val)
    
    # print the first example of the train and val datasets
    print(f"\nTrain dataset length: {len(train_dataset)}")
    print(json.dumps(train_dataset[0], indent=2))
    print(f"Val dataset length: {len(val_dataset)}")
    for i in range(len(val_dataset)):
        print(json.dumps(val_dataset[i], indent=2))
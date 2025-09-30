"""Simple Byte Pair Encoding (BPE) Tokenizer

Logic:
    1. Start with a base vocabulary of all bytes (0-255)
    2. Find the most frequently occurring pair of adjacent tokens
    3. Create a new token that represents this pair
    4. Replace all occurrences of the pair with the new token
    5. Repeat steps 2-4 until desired vocabulary size is reached

Components:
    - get_freqs(): Counts frequency of adjacent token pairs
    - merge(): Replaces all occurrences of a pair with a new token ID
    - BpeTokenizer: Main class that trains and stores merge rules
"""

from typing import Dict, List, Tuple

########################################################
# Helper Functions
########################################################

# find the frequency of each byte pair
def get_freqs(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    Example:
        ids = [1, 2, 3, 1, 2]
        get_freqs(ids) = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    freqs = {}
    # Count how frequently each adjacent id pair appears
    for pair in zip(ids, ids[1:]):
        freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    In a list of integers, merge all consecutive occurrences of pair into a new integer idx.
    Example:
        ids = [1, 2, 3, 1, 2]
        pair = (1, 2)
        idx = 4
        merge(ids, pair, idx) = [4, 3, 4]
    """
    new_ids = []
    # Sweep through ids and replace target pairs with merged token
    i = 0
    while i < len(ids):
        if i<len(ids)-1 and ids[i:i+2] == list(pair):
            # When the next two ids match the merge pair, emit the new token
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


########################################################
# Main BPE Class
########################################################
class BpeTokenizer:
    def __init__(self):
        self.merges = {}
        # Track merges learned during training as pair -> token mapping

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        if verbose:
            print(f"Training BPE tokenizer -> vocab_size: {vocab_size} and num_merges: {num_merges}")
        
        # pre-process input text to bytes
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes) # list of ints
        
        # initialize vocab with single byte representations
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # get frequency of each consecutive pair
            freqs = get_freqs(ids)
            # find the most frequent pair
            pair = max(freqs, key=freqs.get)
            # mint a new token id
            idx = 256 + i
            # replace all the occurences of the pair with the new token id
            ids = merge(ids, pair, idx)
            # store the merge
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # note, here we are concatenating the bytes objects
            self.merges[pair] = idx
            # print the merges
            if verbose:
                print(f"Merge {i+1} / {num_merges}: {pair} -> {idx} ({vocab[idx]} with {freqs[pair]} occurrences)")

# main function (for testing)
if __name__ == "__main__":
    tokenizer = BpeTokenizer()
    # input text file
    with open("data/text.txt", "r") as f:
        text = f.read()
    tokenizer.train(text, 276, verbose=True)

"""Simple Byte Pair Encoding (BPE) Tokenizer

A BPE tokenizer implementation with regex-based text splitting to prevent merges
across category boundaries (e.g., letters, numbers, whitespace).

Training Logic:
    1. Split input text into chunks using regex pattern
    2. Start with base vocabulary of all bytes (0-255)
    3. Find most frequently occurring pair across all chunks
    4. Create new token for that pair
    5. Replace all occurrences with the new token
    6. Repeat steps 3-5 until desired vocabulary size is reached

Components:
    - get_freqs(): Counts frequency of adjacent token pairs
    - merge(): Replaces all occurrences of a pair with a new token ID
    - BpeTokenizer: Main class with train(), encode(), and decode() methods

Note: This implementation does not handle special tokens (e.g., <|endoftext|>).
"""

from typing import Dict, List, Tuple

import regex as re

########################################################
# Helper Functions
########################################################

# find the frequency of each byte pair
def get_freqs(ids: List[int], freqs: Dict[Tuple[int, int], int] = None) -> Dict[Tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    Example:
        ids = [1, 2, 3, 1, 2]
        get_freqs(ids) = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    if freqs is None:
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
# GPT-2 text split patterns: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BpeTokenizer:
    def __init__(self, regex_patterns: str = None):
        self.merges = {} # pair -> token mapping
        self.vocab = {}  # token -> bytes mapping
        self.regex_pattern = GPT2_SPLIT_PATTERN if regex_patterns is None else regex_patterns
        self.compiled_regex_pattern = re.compile(self.regex_pattern) # convert to regular expression object, it is more efficient for reuse
        print(f"Regex pattern: {self.regex_pattern}")
        
    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        if verbose:
            print(f"Training BPE tokenizer -> vocab_size: {vocab_size} and num_merges: {num_merges}")
        
        # pre-process input text to chunks (list of strings)
        if verbose:
            print("Pre-processing input text to chunks...")
        text_chunks = self.compiled_regex_pattern.findall(text)
        
        # convert input text_chunks to list of ints
        ids = [list(t.encode('utf-8')) for t in text_chunks]
        
        # initialize vocab with single byte representations
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # get frequency of each consecutive pair
            freqs = {}
            for chunk_ids in ids:
                freqs = get_freqs(chunk_ids, freqs)
            # find the most frequent pair
            pair = max(freqs, key=freqs.get)
            # mint a new token id
            idx = 256 + i
            # replace all the occurences of the pair with the new token id
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # store the merge
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]] # note, here we are concatenating the bytes objects
            self.merges[pair] = idx
            # print the merges
            if verbose:
                print(f"Merge {i+1} / {num_merges}: {pair} -> {idx} ({self.vocab[idx]} with {freqs[pair]} occurrences)")
    
    def decode(self, ids: List[int]) -> str:
        # errors='replace' replaces any invalid utf-8 bytes with the replacement character
        # see https://docs.python.org/3/library/stdtypes.html#bytes.decode for more details
        return b"".join(self.vocab[id] for id in ids).decode('utf-8', errors='replace')
    
    def _encode_single_chunk_(self, text_bytes: bytes) -> List[int]:
        # string -> list of ints
        ids = list(text_bytes)
        
        while len(ids)>=2:
            # find the pair with the lowest merge index (earliest merge in training)
            # note: since python3.7, the order of dictionary items is guaranteed to be the same as the order of insertion
            freqs = get_freqs(ids)
            pair = min(freqs, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break # as nothing left to merge
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode(self, text: str) -> List[int]:
        # split text into chunks
        text_chunks = self.compiled_regex_pattern.findall(text)
        # encode each chunk
        ids = [self._encode_single_chunk_(t.encode('utf-8')) for t in text_chunks]
        # list of lists of ints -> list of ints
        ids = [idx for chunk_ids in ids for idx in chunk_ids]
        return ids

# main function (for testing)
if __name__ == "__main__":
    # create tokenizer
    tokenizer = BpeTokenizer()
    # input text file
    with open("data/text.txt", "r") as f:
        text = f.read()
    
    # Train tokenizer
    tokenizer.train(text, 276, verbose=True)
    
    # Test examples covering various edge cases
    test_examples = [
        # Basic cases
        "hello world",
        "a",
        "",
        # Repeated patterns (should benefit from BPE merges)
        "aaaaaaa",
        "hello hello hello",
        
        # Unicode and emoji
        "Hello, 世界! 🌍",
        "🚀🌟💻",
        
        # Special characters and punctuation
        "!@#$%^&*()",
        "test...test!!!",

        # Whitespace variations
        "multiple   spaces",
        "tab\tseparated",
        "mixed \t\n whitespace",
        
        # Long repeated sequences
        "hello" * 20,
    ]
    
    print("\n" + "#"*60)
    print("Testing encode/decode on various examples:")
    print("#"*60)
    
    for i, example in enumerate(test_examples, 1):
        # Show truncated version for display if too long
        display_text = example if len(example) <= 40 else example[:37] + "..."
        
        encoded = tokenizer.encode(example)
        decoded = tokenizer.decode(encoded)
        
        # Check roundtrip
        roundtrip_ok = decoded == example
        status = "OK" if roundtrip_ok else "FAIL"
        
        print(f"\n{i}. {status} Text: {repr(display_text)}")
        print(f"   Encoded ({len(encoded)} tokens): {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"   Roundtrip: {'PASS' if roundtrip_ok else 'FAIL'}")
        
        if not roundtrip_ok:
            print(f"   Original:  {repr(example)}")
            print(f"   Decoded:   {repr(decoded)}")
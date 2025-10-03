"""Simple Byte Pair Encoding (BPE) Tokenizer

A BPE tokenizer implementation with regex-based text splitting, special token support, 
and parallel pre-tokenization for efficient training on large datasets.
Uses regex patterns to prevent merges across category boundaries (letters, numbers, whitespace).

Training Logic:
    1. Load input text file and optionally split into chunks using boundary_split_token
    2. Pre-tokenize chunks (sequentially or in parallel using multiprocessing):
       - Split by special tokens (if provided) to exclude them from merges
       - Apply regex pattern splitting to remaining text
       - Convert to byte tuples and track frequencies
    3. Collapse corpus to unique pre-token tuples with occurrence frequencies
       (This optimization avoids redundant processing of repeated tokens)
    4. Start with base vocabulary of all bytes (0-255)
    5. Find most frequently occurring byte pair across unique sequences (weighted by frequencies)
    6. Create new token for that pair and merge in unique sequences only
    7. Repeat until desired vocabulary size is reached

Performance Optimizations:
    1. Frequency-based approach:
       - Converts text chunks to unique id tuples tracked with frequencies using Counter
       - Counts pairs from unique sequences weighted by their occurrence counts
       - Performs merges on unique sequences only, not all occurrences
       - Dramatically reduces computation for datasets with repeated tokens/patterns
    
    2. Parallel pre-tokenization (optional):
       - Splits large files into chunks at boundary_split_token boundaries
       - Processes chunks in parallel using multiprocessing.Pool
       - Merges frequency counts from all chunks
       - Significantly speeds up pre-tokenization stage for large datasets
       - Enable with num_processes parameter (e.g., num_processes=4)

Special Tokens (Two-Step Process):
    1. During Training: Pass special_tokens as List[str] to train() to exclude from regex pattern splitting and BPE merges
       Example: train(input_file_path="data.txt", vocab_size=276, special_tokens=["<|endoftext|>"], 
                     boundary_split_token="<|endoftext|>", num_processes=4)
    
    2. After Training: Register special tokens as Dict[str, int] using register_special_tokens()
       Example: register_special_tokens({"<|endoftext|>": 276})
    
    3. During Encoding: Control behavior with allowed_special_tokens parameter:
       - "all": Encode special tokens as single token IDs
       - "none": Treat special tokens as regular text (apply BPE)
       - "none_raise": Raise ValueError if special tokens found in text

Tokenizer save and load:
    - save(prefix): Saves tokenizer to two files:
        - prefix.model: Binary format with merges and special tokens (used for loading)
        - prefix.vocab: Human-readable vocabulary for inspection
    - load(model_file): Loads a saved tokenizer from .model file

Main Methods:
    - train(input_file_path, vocab_size, special_tokens, boundary_split_token, num_processes, verbose): 
        Train BPE tokenizer on input file with optional parallel processing
    - encode(text, allowed_special_tokens): Convert text to token IDs
    - decode(ids): Convert token IDs back to text
    - save(prefix): Save tokenizer to disk
    - load(model_file): Load tokenizer from disk
    - register_special_tokens(special_tokens): Register special tokens after training
    - get_text_chunks(text, special_tokens, verbose): Split text into chunks using regex
"""

import os
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO, Dict, List, Tuple

import regex as re

########################################################
# Helper Functions
########################################################

# find the frequency of each byte pair across unique id sequences
def get_ids_freqs(ids_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
    Count all adjacent byte pairs across unique id sequences, weighted by their frequencies. This is more efficient than counting pairs from every token occurrence.
    
    Args:
        ids_freqs: Dictionary mapping unique id tuples to their occurrence counts
        
    Returns:
        Dictionary mapping byte pairs to their total occurrence counts
        
    """
    pair_freqs = {}
    for ids, freq in ids_freqs.items():
        for pair in zip(ids, ids[1:]):
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs

def merge(ids: Tuple[int, ...], pair: Tuple[int, int], idx: int) -> Tuple[int, ...]:
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
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            # When the next two ids match the merge pair, emit the new token
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return tuple(new_ids)

def merge_pair_in_ids_freqs(ids_freqs: Dict[Tuple[int, ...], int], pair: Tuple[int, int], idx: int) -> Dict[Tuple[int, ...], int]:
    """
    Merge a specific pair in all unique id sequences and return updated frequencies.
    This only processes unique id sequences, not every occurrence.
    """
    new_ids_freqs = {}
    for ids, freq in ids_freqs.items():
        new_ids = merge(ids, pair, idx)
        new_ids_freqs[new_ids] = freq
    return new_ids_freqs

# helper functions for .vocab file
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def split_text_by_special_tokens(text: str, special_tokens: List[str] = None, verbose=False) -> List[str]:
    """Split text by special tokens."""
    assert special_tokens is not None, "special_tokens must be provided"
    if verbose:
        print(f"Splitting text by special tokens: {special_tokens}")
    
    # Create regex pattern to split by special tokens
    special_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"

    # Split text by special tokens, keeping the special tokens in the result
    text_segments = re.split(special_pattern, text)
    if verbose:
        print(f"Total text segments: {len(text_segments):,}")
    
    # remove empty '' segments and special tokens segments
    text_segments = [segment for segment in text_segments if segment and segment not in special_tokens]
    if verbose:
        print(f"Total text segments after removing empty and special tokens segments: {len(text_segments):,}")
    return text_segments

def process_text_chunk(args: Tuple[str, str, List[str]]) -> Dict[Tuple[int, ...], int]:
    """
    Process a single text chunk for parallel pre-tokenization during training.
    This function must be at module level for multiprocessing pickling.
    
    Args:
        args: Tuple of (chunk_text, regex_pattern, special_tokens)
        
    Returns:
        Counter mapping byte tuples to their frequencies in this chunk
    """
    chunk_text, regex_pattern, special_tokens = args
    compiled_pattern = re.compile(regex_pattern)
    ids_freqs = Counter()
    
    # Split by special tokens if provided
    if special_tokens:
        special_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        text_segments = re.split(special_pattern, chunk_text)
        # Remove empty segments and special tokens
        text_segments = [segment for segment in text_segments if segment and segment not in special_tokens]
        
        # Apply regex pattern to each segment
        for segment in text_segments:
            text_chunks = compiled_pattern.findall(segment)
            for chunk in text_chunks:
                byte_tuple = tuple(chunk.encode('utf-8'))
                ids_freqs[byte_tuple] += 1
    else:
        # No special tokens - just apply regex pattern
        text_chunks = compiled_pattern.findall(chunk_text)
        for chunk in text_chunks:
            byte_tuple = tuple(chunk.encode('utf-8'))
            ids_freqs[byte_tuple] += 1
    
    return ids_freqs

# taken from https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

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
    
        
    def get_text_chunks(self, text: str, special_tokens: List[str] = None, verbose=False) -> List[str]:
        """Split text into chunks using the regex pattern."""
        # Pre-process input text to chunks (list of strings)
        if verbose:
            print("Pre-processing input text to chunks...")
        
        # If special tokens are provided, split text by them first
        if special_tokens:
            text_segments = split_text_by_special_tokens(text, special_tokens, verbose)
            # Process each segment: apply regex pattern to regular text
            text_chunks = []
            for segment in text_segments:
                    text_chunks.extend(self.compiled_regex_pattern.findall(segment))
        else:
            # No special tokens - just apply regex pattern splitting
            text_chunks = self.compiled_regex_pattern.findall(text)
        
        if verbose:
            print(f"Total text chunks: {len(text_chunks)}")
            
        return text_chunks
    
    
    def train(self, input_file_path: str = None, vocab_size: int = 256, special_tokens: List[str] = None, boundary_split_token: str = None, num_processes: int = None, verbose: bool = False):
        """
        Train the BPE tokenizer on the given text or file.
        
        Args:
            input_file_path (str): Path to input text file 
            vocab_size (int): Desired vocabulary size (must be >= 256)
            special_tokens (List[str]): List of special tokens to exclude from merging (e.g., ["<|endoftext|>"])
            boundary_split_token (str): Token to split the text into chunks (e.g., "<|endoftext|>")
            num_processes (int): Number of processes for parallel pre-tokenization (default: 2)
            verbose (bool): Whether to print training progress
        """
        # start time
        train_start = time.time()
        
        assert vocab_size >= 256, "Vocab size must be at least 256"
        assert boundary_split_token is not None, "Boundary split token must be provided, for example, '<|endoftext|>'"
        
        num_merges = vocab_size - 256
        if verbose:
            print(f"Training BPE tokenizer -> vocab_size: {vocab_size} and num_merges: {num_merges}")
        
        # Pre-tokenization stage
        pretokenize_start = time.time()
        if num_processes is not None and num_processes > 1:
            if verbose:
                print("Parallel pre-tokenization")
            # Find chunk boundaries based on special tokens
            num_processes = min(num_processes, os.cpu_count() // 2)
            if verbose:
                print(f"Using {num_processes} processes for parallel pre-tokenization")
            with open(input_file_path, "rb") as f:
                boundaries = find_chunk_boundaries(f, num_processes, boundary_split_token.encode("utf-8"))
            if verbose:
                print(f"Created {len(boundaries)-1} chunks for processing the {input_file_path}")
            
            # Read chunks and prepare for parallel processing
            chunks_args = []
            with open(input_file_path, 'rb') as f:
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
                    chunks_args.append((chunk_text, self.regex_pattern, special_tokens))
                
            # Process chunks in parallel
            if verbose:
                print("Pre-tokenizing chunks in parallel...")
            
            with Pool(num_processes) as pool:
                chunk_results = pool.map(process_text_chunk, chunks_args)
            
            # Combine results from all chunks
            ids_freqs = Counter()
            for chunk_freq in chunk_results:
                ids_freqs.update(chunk_freq)
                
            if verbose:
                print(f"Found {len(ids_freqs):,} unique pre-tokens")
                print(f"Pre-tokenization time: {time.time() - pretokenize_start:.4f} seconds")
        else:
            if verbose:
                print("Sequential pre-tokenization")
            
            text_chunks = self.get_text_chunks(text, special_tokens, verbose)
            
            # Convert text chunks to unique id tuples with frequencies
            if verbose:
                print("Collapsing corpus to unique pre-token tuples with freqs...")
            
            ids_freqs = Counter()
            for chunk in text_chunks:
                # Convert each chunk to a tuple of bytes (makes it hashable for Counter)
                byte_tuple = tuple(chunk.encode('utf-8'))
                ids_freqs[byte_tuple] += 1
            
            if verbose:
                print(f"Pre-tokenization time: {time.time() - pretokenize_start:.4f} seconds")
                print(f"Found {len(ids_freqs):,} unique pre-tokens from {len(text_chunks):,} total chunks")
        
        
        # initialize vocab with single byte representations
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # BPE merges
        for i in range(num_merges):
            merge_start = time.time()
            
            # Count pairs from unique id sequences weighted by their frequencies instead of counting from every token occurrence
            freqs = get_ids_freqs(ids_freqs)
            if not freqs:
                if verbose:
                    print(f"No more pairs to merge after {i} merges")
                break
            
            # find the most frequent pair
            pair = max(freqs, key=freqs.get)
            
            # mint a new token id
            idx = 256 + i
            
            # Merge the pair in unique id sequences only, not all occurrences
            ids_freqs = merge_pair_in_ids_freqs(ids_freqs, pair, idx)
            
            # store the merge
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]] # note, here we are concatenating the bytes objects
            self.merges[pair] = idx
            
            # print the merges
            if verbose:
                print(f"Merge {i+1} / {num_merges}: {pair} -> {idx} ({self.vocab[idx]} with {freqs[pair]} occurrences) in {time.time() - merge_start:.4f} s")
        if verbose:
            print(f"Training time: {(time.time() - train_start)/60:.2f} mins")
    
    # special tokesn = list of dict {key: string, value: int}
    # replica of minbpe
    def register_special_tokens(self, special_tokens: Dict[str, int]):
        self.special_tokens = special_tokens
        print(f"Registered special tokens: {special_tokens}")
        self.inv_special_tokens = {v: k for k, v in special_tokens.items()}
        
    def decode(self, ids: List[int]) -> str:
        # errors='replace' replaces any invalid utf-8 bytes with the replacement character
        # see https://docs.python.org/3/library/stdtypes.html#bytes.decode for more details
        input_bytes = []
        for idx in ids:
            if idx in self.vocab:
                input_bytes.append(self.vocab[idx])
            elif idx in self.inv_special_tokens:
                input_bytes.append(self.inv_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        return b"".join(input_bytes).decode('utf-8', errors='replace')
    
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
    
    def _encode_no_special_tokens(self, text: str) -> List[int]:
        # split text into chunks
        text_chunks = self.compiled_regex_pattern.findall(text)
        # encode each chunk
        ids = [self._encode_single_chunk_(t.encode('utf-8')) for t in text_chunks]
        # list of lists of ints -> list of ints
        ids = [idx for chunk_ids in ids for idx in chunk_ids]
        return ids
    
    def encode(self, text: str, allowed_special_tokens: str = "none_raise") -> List[int]:
        """
        Encode text into a list of token IDs using BPE tokenization with regex pattern splitting
        and special token handling.
        
        Args:
            text (str): The input text to encode
            allowed_special_tokens (str): Policy for handling special tokens:
                - "none": Ignore special tokens, encode them as regular text
                - "all": Allow all registered special tokens to be encoded as single tokens
                - "none_raise": Raise error if any special tokens are found in text
        """
        # Validate the allowed_special_tokens parameter
        assert allowed_special_tokens in ["all", "none", "none_raise"], "allowed_special_tokens must be 'all'/'none'/'none_raise'"
        
        # Convert string policy to actual token dictionary
        if allowed_special_tokens == "none":
            # Treat special tokens as regular text
            allowed_special_tokens = {}
        elif allowed_special_tokens == "all":
            # Use all registered special tokens
            allowed_special_tokens = self.special_tokens
        elif allowed_special_tokens == "none_raise":
            # Don't allow special tokens, but raise error if found
            allowed_special_tokens = {}
            assert all(token not in text for token in self.special_tokens), "Special tokens found in text"
        
        # If no special tokens are allowed, use the simpler encoding path
        if not allowed_special_tokens:
            return self._encode_no_special_tokens(text)
        
        # Create regex pattern to split text around special tokens
        # The parentheses in the pattern ensure special tokens are included in the split results
        # re.escape prevents regex injection from special token strings
        special_pattern = "(" + "|".join(re.escape(k) for k in allowed_special_tokens) + ")"    
        all_chunks = re.split(special_pattern, text)
        
        # Process each chunk: encode special tokens directly, encode regular text with BPE
        ids = [] 
        for chunk in all_chunks:
            if chunk in allowed_special_tokens:
                # chunk is a special token - add its ID directly
                ids.append(allowed_special_tokens[chunk])
            else:
                # chunk is regular text - apply BPE encoding with regex pattern splitting
                ids.extend(self._encode_no_special_tokens(chunk))
        return ids
    
    # taken and modified from https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("simple-bpe v1\n")
            f.write(f"{self.regex_pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")
            
            # write special tokens at the end for human readability
            for special, idx in self.special_tokens.items():
                f.write(f"[{special}] {idx}\n")

    # taken and modified from https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        return vocab
    
    # taken and modified from https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "simple-bpe v1"
            # read the pattern
            self.regex_pattern = f.readline().strip()
            # read the special tokenså
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.inv_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()
        
        
# main function (for testing)
if __name__ == "__main__":
    # create tokenizer
    tokenizer = BpeTokenizer()
    # input text file
    with open("data/text.txt", "r") as f:
        text = f.read()
    
    # Train tokenizer with special tokens
    # Note: We pass special_tokens during training to exclude them from BPE merges
    special_token_list = ["<|endoftext|>"]
    tokenizer.train(text, 276, special_tokens=special_token_list, verbose=True)
    # Register special tokens with their IDs for encoding/decoding
    tokenizer.register_special_tokens({"<|endoftext|>": 276})
    tokenizer.save("data/tok276")
    
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
        
        # Special tokens
        "<|endoftext|>",
        "<|endoftext|>hello world<|endoftext|>",
        "<|endoftext|>hello world<|endoftext|>ojdf[h[fsdh [fjsd[jfs[a]]]]]",
    ]
    
    print("\n" + "#"*60)
    print("Testing encode/decode on various examples:")
    print("#"*60)
    
    for i, example in enumerate(test_examples, 1):
        # Show truncated version for display if too long
        display_text = example if len(example) <= 40 else example[:37] + "..."
        
        encoded = tokenizer.encode(example, allowed_special_tokens="all")
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
    
    # load the tokenizer
    del tokenizer
    tokenizer = BpeTokenizer()
    tokenizer.load("data/tok276.model")
    for i, example in enumerate(test_examples, 1):
        encoded = tokenizer.encode(example, allowed_special_tokens="all")
        decoded = tokenizer.decode(encoded)
        roundtrip_ok = decoded == example
        status = "OK" if roundtrip_ok else "FAIL"
        print(f"\n{i}. {status} Text: {repr(example)}")
        print(f"   Encoded ({len(encoded)} tokens): {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"   Roundtrip: {'PASS' if roundtrip_ok else 'FAIL'}")
        if not roundtrip_ok:
            print(f"   Original:  {repr(example)}")
            print(f"   Decoded:   {repr(decoded)}")
"""Benchmark encoding performance of BPE tokenizer on TinyStoriesV2 validation set"""

import sys
import time

from bpe import BpeTokenizer

# Try to import tiktoken (optional)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Try to import Rust implementation (optional)
try:
    from bpe_encode_rust import PyBpeEncode
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def benchmark_encoding(model_path, input_file_path, allowed_special="none_raise", use_tiktoken=False, tiktoken_model="gpt2", use_rust=False):
    """
    Benchmark tokenizer encoding performance.
    
    Args:
        model_path: Path to the .model file
        input_file_path: Path to the text file to encode
        allowed_special: Special token handling policy
        use_tiktoken: If True, use tiktoken instead of BpeTokenizer
        tiktoken_model: Model name for tiktoken (e.g., "gpt2", "cl100k_base")
        use_rust: If True, use Rust implementation (bpe_encode_rust)
    """
    print("="*70)
    if use_tiktoken:
        print(f"Tiktoken Tokenizer Encoding Benchmark ({tiktoken_model})")
    elif use_rust:
        print("Rust BPE Tokenizer Encoding Benchmark")
    else:
        print("Python BPE Tokenizer Encoding Benchmark")
    print("="*70)
    
    # Load tokenizer
    load_start = time.time()
    
    if use_tiktoken:
        if not TIKTOKEN_AVAILABLE:
            print("❌ Error: tiktoken is not installed. Install with: pip install tiktoken")
            sys.exit(1)
        print(f"\n📥 Loading tiktoken model: {tiktoken_model}")
        tokenizer = tiktoken.get_encoding(tiktoken_model)
        load_time = time.time() - load_start
        print(f"✓ Tokenizer loaded in {load_time:.4f} seconds")
        print(f"  Model: {tiktoken_model}")
        print(f"  Vocab size: {tokenizer.n_vocab:,}")
    elif use_rust:
        if not RUST_AVAILABLE:
            print("❌ Error: Rust implementation not found.")
            print("Build with: cd encode_rs && maturin develop --release --features python")
            sys.exit(1)
        print(f"\n📥 Loading Rust tokenizer from: {model_path}")
        tokenizer = PyBpeEncode()
        tokenizer.load(model_path)
        load_time = time.time() - load_start
        print(f"✓ Tokenizer loaded in {load_time:.4f} seconds")
        # Rust implementation doesn't expose vocab/merges count directly
    else:
        print(f"\n📥 Loading Python tokenizer from: {model_path}")
        tokenizer = BpeTokenizer()
        tokenizer.load(model_path)
        load_time = time.time() - load_start
        print(f"✓ Tokenizer loaded in {load_time:.4f} seconds")
        print(f"  Vocab size: {len(tokenizer.vocab):,}")
        print(f"  Merges: {len(tokenizer.merges):,}")
        print(f"  Special tokens: {len(tokenizer.special_tokens):,}")
    
    # Read input file
    print(f"\n📖 Reading input file: {input_file_path}")
    read_start = time.time()
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    read_time = time.time() - read_start
    
    num_chars = len(text)
    num_bytes = len(text.encode('utf-8'))
    
    print(f"✓ File read in {read_time:.4f} seconds")
    print(f"  Characters: {num_chars:,}")
    print(f"  Bytes (UTF-8): {num_bytes:,} ({format_size(num_bytes)})")
    print(f"  Lines: {text.count(chr(10)):,}")
    
    # Encode text
    if use_tiktoken:
        # tiktoken uses different parameter: allowed_special (set or "all")
        if allowed_special == "all":
            tiktoken_allowed = "all"
        else:
            tiktoken_allowed = set()  # No special tokens
        print(f"\n🔄 Encoding text (allowed_special={tiktoken_allowed})...")
        encode_start = time.time()
        tokens = tokenizer.encode(text, allowed_special=tiktoken_allowed)
        encode_time = time.time() - encode_start
    elif use_rust:
        # Rust implementation uses string parameter like "all", "none", "none_raise"
        print(f"\n🔄 Encoding text (allowed_special='{allowed_special}')...")
        encode_start = time.time()
        tokens = tokenizer.encode(text, allowed_special)
        encode_time = time.time() - encode_start
    else:
        print(f"\n🔄 Encoding text (allowed_special='{allowed_special}')...")
        encode_start = time.time()
        tokens = tokenizer.encode(text, allowed_special_tokens=allowed_special)
        encode_time = time.time() - encode_start
    
    num_tokens = len(tokens)
    
    print(f"✓ Encoding completed in {encode_time:.4f} seconds")
    print(f"  Tokens: {num_tokens:,}")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("Performance Metrics")
    print("="*70)
    
    chars_per_sec = num_chars / encode_time
    bytes_per_sec = num_bytes / encode_time
    tokens_per_sec = num_tokens / encode_time
    
    print("\n📊 Throughput:")
    print(f"  {chars_per_sec:,.0f} chars/sec")
    print(f"  {format_size(bytes_per_sec)}/sec")
    print(f"  {tokens_per_sec:,.0f} tokens/sec")
    
    print("\n📈 Compression:")
    print(f"  {num_chars / num_tokens:.2f} chars/token")
    print(f"  {num_bytes / num_tokens:.2f} bytes/token")
    
    print("\n⏱️  Timing Breakdown:")
    print(f"  Load tokenizer: {load_time:.4f}s")
    print(f"  Read file:      {read_time:.4f}s")
    print(f"  Encode:         {encode_time:.4f}s")
    print(f"  Total:          {load_time + read_time + encode_time:.4f}s")
    
    print("\n" + "="*70)
    
    return {
        'num_chars': num_chars,
        'num_bytes': num_bytes,
        'num_tokens': num_tokens,
        'encode_time': encode_time,
        'chars_per_sec': chars_per_sec,
        'bytes_per_sec': bytes_per_sec,
        'tokens_per_sec': tokens_per_sec
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark BPE (Python/Rust) or tiktoken encoding performance")
    parser.add_argument("--model", default="data/tinystoriesv2-gpt4-train-2048.model",
                        help="Path to BPE model file (default: data/tinystoriesv2-gpt4-train-2048.model)")
    parser.add_argument("--input", default="data/TinyStoriesV2-GPT4-valid.txt",
                        help="Path to input text file (default: data/TinyStoriesV2-GPT4-valid.txt)")
    parser.add_argument("--tiktoken", action="store_true",
                        help="Use tiktoken instead of BpeTokenizer")
    parser.add_argument("--tiktoken-model", default="gpt2",
                        help="Tiktoken model name (default: gpt2). Options: gpt2, cl100k_base, o200k_base")
    parser.add_argument("--rust", action="store_true",
                        help="Use Rust implementation (bpe_encode_rust)")
    parser.add_argument("--special", default="all",
                        help="Special token handling for BPE (default: all). Options: all, none, none_raise")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_encoding(
        args.model, 
        args.input, 
        allowed_special=args.special,
        use_tiktoken=args.tiktoken,
        tiktoken_model=args.tiktoken_model,
        use_rust=args.rust
    )
    
    # Optional: Test a smaller sample for quick verification
    if args.rust:
        print("\n" + "#"*70)
        print("Quick Verification Test (first 1000 chars)")
        print("#"*70)
        
        with open(args.input, 'r', encoding='utf-8') as f:
            sample = f.read(1000)
        
        tokenizer = PyBpeEncode()
        tokenizer.load(args.model)
        
        tokens = tokenizer.encode(sample, args.special)
        decoded = tokenizer.decode(tokens)
        
        print(f"Original:  {repr(sample[:100])}...")
        print(f"Tokens:    {tokens[:20]}...")
        print(f"Decoded:   {repr(decoded[:100])}...")
        print(f"Roundtrip: {'✓ PASS' if sample == decoded else '✗ FAIL'}")
    elif not args.tiktoken:
        print("\n" + "#"*70)
        print("Quick Verification Test (first 1000 chars)")
        print("#"*70)
        
        with open(args.input, 'r', encoding='utf-8') as f:
            sample = f.read(1000)
        
        tokenizer = BpeTokenizer()
        tokenizer.load(args.model)
        
        tokens = tokenizer.encode(sample, allowed_special_tokens=args.special)
        decoded = tokenizer.decode(tokens)
        
        print(f"Original:  {repr(sample[:100])}...")
        print(f"Tokens:    {tokens[:20]}...")
        print(f"Decoded:   {repr(decoded[:100])}...")
        print(f"Roundtrip: {'✓ PASS' if sample == decoded else '✗ FAIL'}")
    else:
        # Simple verification for tiktoken
        print("\n" + "#"*70)
        print("Quick Verification Test (first 1000 chars)")
        print("#"*70)
        
        with open(args.input, 'r', encoding='utf-8') as f:
            sample = f.read(1000)
        
        tokenizer = tiktoken.get_encoding(args.tiktoken_model)
        
        # Handle special tokens same way as main benchmark
        if args.special == "all":
            tiktoken_allowed = "all"
        else:
            tiktoken_allowed = set()
        
        tokens = tokenizer.encode(sample, allowed_special=tiktoken_allowed)
        decoded = tokenizer.decode(tokens)
        
        print(f"Original:  {repr(sample[:100])}...")
        print(f"Tokens:    {tokens[:20]}...")
        print(f"Decoded:   {repr(decoded[:100])}...")
        print(f"Roundtrip: {'✓ PASS' if sample == decoded else '✗ FAIL'}")

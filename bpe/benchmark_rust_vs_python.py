"""Benchmark Rust vs Python BPE encoding performance.

This script compares the encoding performance of the Python implementation
against the Rust implementation (if built with Python bindings).

Usage:
    # First build the Rust Python bindings:
    cd encode_rs
    maturin develop --release --features python
    cd ..
    
    # Then run the benchmark:
    python benchmark_rust_vs_python.py
"""

import time
from pathlib import Path

# Python implementation
from bpe import BpeTokenizer


def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

# Try to import Rust implementation
try:
    from bpe_encode_rust import PyBpeEncode
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("WARNING: Rust implementation not found.")
    print("Build with: cd encode_rs && maturin develop --release --features python")
    print()


def benchmark_encoding(encoder, text: str, name: str, iterations: int = 10, is_rust: bool = False) -> tuple:
    """Benchmark encoding performance."""
    times = []
    
    # Calculate text metrics
    num_chars = len(text)
    num_bytes = len(text.encode('utf-8'))
    
    # Warmup
    for _ in range(3):
        if is_rust:
            _ = encoder.encode(text, "all")
        else:
            _ = encoder.encode(text, allowed_special_tokens="all")
    
    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        if is_rust:
            ids = encoder.encode(text, "all")
        else:
            ids = encoder.encode(text, allowed_special_tokens="all")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate throughput metrics
    chars_per_sec = num_chars / avg_time
    bytes_per_sec = num_bytes / avg_time
    tokens_per_sec = len(ids) / avg_time
    
    print(f"{name}:")
    print(f"  Time:       {avg_time*1000:.2f}ms (min: {min_time*1000:.2f}ms, max: {max_time*1000:.2f}ms)")
    print(f"  Throughput: {chars_per_sec:,.0f} chars/sec | {format_size(bytes_per_sec)}/sec | {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Tokens:     {len(ids):,}")
    print()
    
    return avg_time, ids, chars_per_sec, bytes_per_sec


def main():
    # Configuration
    model_path = "data/tinystoriesv2-gpt4-train-2048.model"
    data_path = "data/TinyStoriesV2-GPT4-valid.txt"
    iterations = 5
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        print("Using sample text instead...")
        text = "The quick brown fox jumps over the lazy dog. " * 100
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    num_chars = len(text)
    num_bytes = len(text.encode('utf-8'))
    
    print("Benchmark Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Text: {num_chars:,} characters | {format_size(num_bytes)}")
    print(f"  Iterations: {iterations}")
    print()
    
    # Load Python encoder
    print("Loading Python encoder...")
    py_encoder = BpeTokenizer()
    py_encoder.load(model_path)
    print()
    
    # Benchmark Python
    print("=" * 60)
    print("PYTHON IMPLEMENTATION")
    print("=" * 60)
    py_time, py_ids, py_chars_per_sec, py_bytes_per_sec = benchmark_encoding(py_encoder, text, "Python BPE", iterations, is_rust=False)
    
    if RUST_AVAILABLE:
        # Load Rust encoder
        print("=" * 60)
        print("RUST IMPLEMENTATION")
        print("=" * 60)
        print("Loading Rust encoder...")
        rust_encoder = PyBpeEncode()
        rust_encoder.load(model_path)
        print()
        
        # Benchmark Rust
        rust_time, rust_ids, rust_chars_per_sec, rust_bytes_per_sec = benchmark_encoding(rust_encoder, text, "Rust BPE", iterations, is_rust=True)
        
        # Compare
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        speedup = py_time / rust_time
        throughput_speedup = rust_chars_per_sec / py_chars_per_sec
        print(f"⚡ Speed:      {speedup:.2f}x faster with Rust")
        print(f"⏱️  Time saved: {(py_time - rust_time)*1000:.2f}ms per encode")
        print("📊 Throughput:")
        print(f"   Python: {py_chars_per_sec:,.0f} chars/sec | {format_size(py_bytes_per_sec)}/sec")
        print(f"   Rust:   {rust_chars_per_sec:,.0f} chars/sec | {format_size(rust_bytes_per_sec)}/sec")
        print(f"   Speedup: {throughput_speedup:.2f}x")
        print()
        
        # Verify correctness on smaller sample
        print("Verifying correctness (first 1000 chars)...")
        test_sample = text[:1000]
        py_test_ids = py_encoder.encode(test_sample, allowed_special_tokens="none")
        rust_test_ids = rust_encoder.encode(test_sample, "none")
        
        if py_test_ids == rust_test_ids:
            print("✓ Token IDs match!")
            # Also verify decoding
            py_decoded = py_encoder.decode(py_test_ids)
            rust_decoded = rust_encoder.decode(rust_test_ids)
            if py_decoded == rust_decoded:
                print("✓ Decoded text matches!")
        else:
            print("✗ Token IDs differ!")
            print(f"Python IDs (first 20): {py_test_ids[:20]}")
            print(f"Rust IDs (first 20):   {rust_test_ids[:20]}")
    else:
        print("=" * 60)
        print("RUST IMPLEMENTATION NOT AVAILABLE")
        print("=" * 60)
        print("To build Rust bindings:")
        print("  1. Install maturin: pip install maturin")
        print("  2. Navigate to encode_rs: cd encode_rs")
        print("  3. Build: maturin develop --release --features python")
        print("  4. Return to bpe: cd ..")
        print()


if __name__ == "__main__":
    main()


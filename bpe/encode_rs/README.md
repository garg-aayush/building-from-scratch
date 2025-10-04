# BPE Encoder - Rust Implementation

This is a partially vibe-coded Rust implementation of Byte Pair Encoding (BPE) **encode/decode inference** with Python bindings. This provides **3-4x faster** encoding compared to the pure Python implementation while maintaining full compatibility (as tested on the [TinyStoriesV2 validation dataset](https://huggingface.co/datasets/roneneldan/TinyStoriesV2/blob/main/TinyStoriesV2-GPT4-valid.txt) on Macbook Pro M3 Max (36 GB)).

**Note:** This implementation is for encoding and decoding text using pre-trained BPE models only. It does not include training functionality.

The implementation is designed to be a drop-in replacement for the Python version (`bpe.py`) with identical results but significantly better performance.

## Performance Comparison
Benchmarked on TinyStoriesV2 validation set (21.5 MB, 22.4M characters):

| Implementation | Time | Throughput | Speedup |
|----------------|------|------------|---------|
| **Python** | 19.4 sec | 1.16 MB/s | 1.0x |
| **Rust (via Python)** | 5.1 sec | 4.40 MB/s | **3.79x** |


## Why did I write this?

My Python BPE encoding was slow for large texts (~1.1 MB/s). I wanted to create a faster BPE encoder for large texts, and one of the most straightforward ways to do that is to write a Rust/C implementation. I went ahead with Rust because it provides near-C performance with memory safety and can be called from Python with minimal overhead. To my pleasant surprise, this resulted in a **3.79x speedup** on the test dataset (22+ MB encoded in ~5 seconds).

## How did I build this?

I'm not proficient at coding in Rust. I can read and understand Rust code and know how to run and use it, but writing it from scratch is not my cup of tea. So I "vibe-coded" it. Here's how I did it:   

1. Started with my simple Python `BpeTokenizer` class in `bpe.py` with only encode, load, and decode functions
2. Asked `claude-4.5-sonnet` to write a 1-to-1 Rust implementation with specific constraints (same API, same file format compatibility)
3. Wrote a simple `run.rs` test file to verify the implementation
4. Iteratively tested, identified bugs (especially the vocabulary build order issue), and had Claude help fix them
5. Added Python bindings via PyO3 to make it usable from Python

The result is a working, fast implementation that I can maintain and understand, even without deep Rust expertise.


## Folder Structure

```
encode_rs/
├── README.md              # This file
├── Cargo.toml            # Rust package configuration
├── Cargo.lock            # Dependency lockfile (for reproducible builds)
├── run.rs                # Simple Rust test script for encode/decode
├── run.py                # Simple Python test script (uses Rust bindings)
└── src/
    ├── lib.rs           # Library entry point + Python bindings
    └── bpe_encode.rs    # Core BPE encode/decode implementation
```

## Quick Start

### Prerequisites

- **Rust**: Install from https://rustup.rs/
- **Python 3.7+** (if using Python bindings)
- **maturin**: For building Python bindings (`pip install maturin`)

### Option 1: Use from Rust

```bash
# Build the library
cd encode_rs
cargo build --release

# Run the test program
cargo run --release --bin run
```
You can write your own Rust code to use the BPE encoder by importing the `bpe_encode` module similar to the example in `run.rs`.

### Option 2: Use from Python (My preferred method)

This gives you Rust speed with Python convenience.

```bash
# Build Python bindings
cd encode_rs
pip install maturin
maturin develop --release --features python

# Test it
python run.py
```
Similar to the Rust version, you can write your own Python code to use the BPE encoder by importing the `bpe_encode_rust` module similar to the example in `run.py`.

### Benchmark
Once you have everything installed, you can benchmark the performance of the Rust implementation and verify correctness by running:

```bash
# From bpe/ directory
python benchmark_rust_vs_python.py
```

This script not only measures performance but also verifies that both implementations produce identical outputs.

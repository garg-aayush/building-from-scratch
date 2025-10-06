"""Test BPE Encoder - Rust Python Bindings

This script tests the Rust implementation of BPE encoder through Python bindings.

Usage:
    python run.py
    
Make sure you've built the Python bindings first:
    cd encode_rs
    maturin develop --release --features python
"""

from bpe_encode_rust import PyBpeEncode


def main():
    print("Test BPE Encoder - Python Bindings\n")
    
    # Initialize encoder
    encoder = PyBpeEncode()
    
    # Load trained model
    print("Loading model...")
    encoder.load("../data/tok276.model")
    print()
    
    # Test 1: Simple encoding
    print("Test 1: Basic Encoding")
    text = "Hello, world!"
    print(f"Input:  {text!r}")
    
    # All options: "none", "none_raise", "all"
    ids = encoder.encode(text, "none")
    print(f"Tokens: {ids}")
    print(f"Count:  {len(ids)} tokens\n")
    
    # Test 2: Decoding
    print("Test 2: Decoding")
    decoded = encoder.decode(ids)
    print(f"Output: {decoded!r}\n")
    
    # Test 3: Roundtrip verification
    print("Test 3: Roundtrip Verification")
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original: {test_text!r}")
    
    encoded = encoder.encode(test_text, "none")
    roundtrip = encoder.decode(encoded)
    print(f"Decoded:  {roundtrip!r}")
    
    if test_text == roundtrip:
        print("✓ Roundtrip PASSED\n")
    else:
        print("✗ Roundtrip FAILED\n")
    
    # Test 4: Compression stats
    print("Test 4: Compression Stats")
    long_text = "Attention is all you need is one of the most important papers in the field of machine learning in the last decade."
    print(f"Text length: {len(long_text)} chars")
    
    tokens = encoder.encode(long_text, "none")
    print(f"Token count: {len(tokens)}")
    print(f"Compression: {len(long_text) / len(tokens):.2f}x\n")
    
    print("All tests completed!")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("❌ Error: Could not import bpe_encode_rust")
        print("\nMake sure you've built the Python bindings:")
        print("  cd encode_rs")
        print("  maturin develop --release --features python")
        print(f"\nDetails: {e}")
    except FileNotFoundError as e:
        print("❌ Error: Model file not found")
        print(f"Details: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")



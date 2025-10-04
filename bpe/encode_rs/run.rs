use bpe_encode::{AllowedSpecialTokens, BpeEncode};

fn main() {
    println!("Test BPE Encoder\n");
    
    // Initialize encoder
    let mut encoder = BpeEncode::new();
    
    // Load trained model
    println!("Loading model...");
    encoder.load("../data/tok276.model").unwrap();
    println!();
    
    // Test 1: Simple encoding
    println!("Test 1: Basic Encoding");
    let text = "Hello, world!";
    println!("Input:  {:?}", text);
    
    // All options: AllowedSpecialTokens::None, AllowedSpecialTokens::NoneRaise, AllowedSpecialTokens::All
    let ids = encoder.encode(text, AllowedSpecialTokens::None).unwrap();
    println!("Tokens: {:?}", ids);
    println!("Count:  {} tokens\n", ids.len());
    
    // Test 2: Decoding
    println!("Test 2: Decoding");
    let decoded = encoder.decode(&ids).unwrap();
    println!("Output: {:?}\n", decoded);
    
    // Test 3: Roundtrip verification
    println!("Test 3: Roundtrip Verification");
    let test_text = "The quick brown fox jumps over the lazy dog.";
    println!("Original: {:?}", test_text);
    
    let encoded = encoder.encode(test_text, AllowedSpecialTokens::None).unwrap();
    let roundtrip = encoder.decode(&encoded).unwrap();
    println!("Decoded:  {:?}", roundtrip);
    
    if test_text == roundtrip {
        println!("✓ Roundtrip PASSED\n");
    } else {
        println!("✗ Roundtrip FAILED\n");
    }
    
    // Test 4: Compression stats
    println!("Test 4: Compression Stats");
    let long_text = "Attention is all you need is one of the most important papers in the field of machine learning in the last decade.";
    println!("Text length: {} chars", long_text.len());
    
    let tokens = encoder.encode(long_text, AllowedSpecialTokens::None).unwrap();
    println!("Token count: {}", tokens.len());
    println!("Compression: {:.2}x\n", long_text.len() as f64 / tokens.len() as f64);
    
    println!("All tests completed!");
}


//! Lightweight BPE Encoder
//!
//! A simplified encoder class that loads a trained BPE model and provides
//! encoding functionality. This is useful when you only need to encode text
//! and don't need training or other tokenizer features.
//!
//! Main Methods:
//!     - load(model_file): Load a trained BPE model from .model file
//!     - encode(text, allowed_special_tokens): Encode text to token IDs
//!     - decode(ids): Decode token IDs back to text

use fancy_regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Merge all consecutive occurrences of a byte pair into a new token.
///
/// # Arguments
/// * `ids` - Slice of token IDs
/// * `pair` - The byte pair to merge (tuple of two u32s)
/// * `idx` - The new token ID to replace the pair with
///
/// # Returns
/// Vector of token IDs with pairs merged
///
/// # Example
/// ```ignore
/// let ids = vec![1, 2, 3, 1, 2];
/// let pair = (1, 2);
/// let idx = 4;
/// let result = merge(&ids, pair, idx);
/// assert_eq!(result, vec![4, 3, 4]);
/// ```
fn merge(ids: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;
    
    while i < ids.len() {
        if i < ids.len() - 1 && (ids[i], ids[i + 1]) == pair {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    
    new_ids
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowedSpecialTokens {
    None,
    All,
    NoneRaise,
}

impl AllowedSpecialTokens {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(AllowedSpecialTokens::None),
            "all" => Ok(AllowedSpecialTokens::All),
            "none_raise" => Ok(AllowedSpecialTokens::NoneRaise),
            _ => Err(format!("Invalid allowed_special_tokens value: {}", s)),
        }
    }
}

/// Lightweight BPE encoder for encoding text using a pre-trained model.
pub struct BpeEncode {
    /// Merge rules: (token_id1, token_id2) -> merged_token_id
    merges: HashMap<(u32, u32), u32>,
    /// Vocabulary: token_id -> bytes
    vocab: HashMap<u32, Vec<u8>>,
    /// Special tokens: special_token_str -> token_id
    special_tokens: HashMap<String, u32>,
    /// Inverse special tokens: token_id -> special_token_str
    inv_special_tokens: HashMap<u32, String>,
    /// Regex pattern for text splitting
    regex_pattern: String,
    /// Compiled regex pattern
    compiled_regex: Option<Regex>,
}

impl BpeEncode {
    /// Initialize an empty BPE encoder. Use load() to load a trained model.
    pub fn new() -> Self {
        BpeEncode {
            merges: HashMap::new(),
            vocab: HashMap::new(),
            special_tokens: HashMap::new(),
            inv_special_tokens: HashMap::new(),
            regex_pattern: String::new(),
            compiled_regex: None,
        }
    }

    /// Load a trained BPE model from a .model file.
    ///
    /// # Arguments
    /// * `model_file` - Path to the .model file (should end with .model)
    ///
    /// # Model File Format
    /// - Line 1: Version string (e.g., "simple-bpe v1")
    /// - Line 2: Regex pattern for text splitting
    /// - Line 3: Number of special tokens
    /// - Next N lines: Special token and its ID
    /// - Remaining lines: Merge pairs (two token IDs per line)
    pub fn load(&mut self, model_file: &str) -> Result<(), String> {
        if !model_file.ends_with(".model") {
            return Err("Model file must end with .model".to_string());
        }

        let file = File::open(model_file)
            .map_err(|e| format!("Failed to open model file: {}", e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read version
        let version = lines
            .next()
            .ok_or("Empty model file")?
            .map_err(|e| e.to_string())?
            .trim()
            .to_string();
        if version != "simple-bpe v1" {
            return Err(format!("Unsupported version: {}", version));
        }

        // Read regex pattern
        self.regex_pattern = lines
            .next()
            .ok_or("Missing regex pattern")?
            .map_err(|e| e.to_string())?
            .trim()
            .to_string();
        self.compiled_regex = Some(
            Regex::new(&self.regex_pattern)
                .map_err(|e| format!("Invalid regex pattern: {}", e))?,
        );

        // Read special tokens
        let num_special: usize = lines
            .next()
            .ok_or("Missing special token count")?
            .map_err(|e| e.to_string())?
            .trim()
            .parse()
            .map_err(|e| format!("Invalid special token count: {}", e))?;

        for _ in 0..num_special {
            let line = lines.next().ok_or("Missing special token line")?
                .map_err(|e| e.to_string())?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!("Invalid special token line: {}", line));
            }
            let special = parts[0].to_string();
            let special_idx: u32 = parts[1]
                .parse()
                .map_err(|e| format!("Invalid special token ID: {}", e))?;
            self.special_tokens.insert(special.clone(), special_idx);
            self.inv_special_tokens.insert(special_idx, special);
        }

        // Read merges
        let mut idx = 256u32; // Start after base byte tokens (0-255)
        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() != 2 {
                continue; // Skip invalid lines
            }
            let idx1: u32 = parts[0]
                .parse()
                .map_err(|e| format!("Invalid merge token ID: {}", e))?;
            let idx2: u32 = parts[1]
                .parse()
                .map_err(|e| format!("Invalid merge token ID: {}", e))?;
            self.merges.insert((idx1, idx2), idx);
            idx += 1;
        }

        // Build vocabulary
        self.vocab = self.build_vocab();

        println!("Loaded BPE model from {}", model_file);
        println!("  Vocabulary size: {}", self.vocab.len());
        println!("  Number of merges: {}", self.merges.len());
        println!(
            "  Special tokens: {:?}",
            self.special_tokens.keys().collect::<Vec<_>>()
        );
        println!("  Regex pattern: {}", self.regex_pattern);

        Ok(())
    }

    /// Build vocabulary from merges.
    ///
    /// Starts with base 256 byte tokens, then constructs merged tokens
    /// by concatenating the bytes of their constituent tokens.
    ///
    /// # Returns
    /// HashMap mapping token IDs to their byte representations
    fn build_vocab(&self) -> HashMap<u32, Vec<u8>> {
        let mut vocab = HashMap::new();

        // Base 256 byte tokens
        for idx in 0..256u32 {
            vocab.insert(idx, vec![idx as u8]);
        }

        // Build merged tokens IN ORDER
        // We need to sort by the merge index because later merges may depend on earlier ones
        let mut merge_list: Vec<_> = self.merges.iter().collect();
        merge_list.sort_by_key(|(_, &idx)| idx);
        
        for (pair, idx) in merge_list {
            let mut bytes = Vec::new();
            if let Some(b1) = vocab.get(&pair.0) {
                bytes.extend_from_slice(b1);
            }
            if let Some(b2) = vocab.get(&pair.1) {
                bytes.extend_from_slice(b2);
            }
            vocab.insert(*idx, bytes);
        }

        vocab
    }

    /// Encode a single text chunk (no regex splitting or special tokens).
    ///
    /// # Arguments
    /// * `text_bytes` - Bytes to encode
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Algorithm
    /// 1. Start with individual bytes as token IDs
    /// 2. Repeatedly find the pair that was merged earliest in training
    /// 3. Merge that pair into a single token
    /// 4. Continue until no more mergeable pairs exist
    fn encode_single_chunk(&self, text_bytes: &[u8]) -> Vec<u32> {
        // Start with individual bytes as u32
        let mut ids: Vec<u32> = text_bytes.iter().map(|&b| b as u32).collect();

        // Repeatedly merge pairs
        while ids.len() >= 2 {
            // Find all unique pairs in current sequence
            let mut pairs = HashMap::new();
            for i in 0..ids.len() - 1 {
                let pair = (ids[i], ids[i + 1]);
                pairs.entry(pair).or_insert(());
            }

            // Find the pair with lowest merge index (earliest in training)
            let mut best_pair = None;
            let mut best_idx = u32::MAX;

            for pair in pairs.keys() {
                if let Some(&merge_idx) = self.merges.get(pair) {
                    if merge_idx < best_idx {
                        best_idx = merge_idx;
                        best_pair = Some(*pair);
                    }
                }
            }

            // If no pairs are in our merge dict, we're done
            if best_pair.is_none() {
                break;
            }

            // Merge this pair
            let pair = best_pair.unwrap();
            let idx = self.merges[&pair];
            ids = merge(&ids, pair, idx);
        }

        ids
    }

    /// Encode text without handling special tokens.
    ///
    /// # Arguments
    /// * `text` - Text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Process
    /// 1. Split text into chunks using regex pattern
    /// 2. Encode each chunk separately
    /// 3. Flatten results into single vector
    fn encode_no_special_tokens(&self, text: &str) -> Vec<u32> {
        let regex = self.compiled_regex.as_ref().unwrap();

        // Split text into chunks using regex
        let mut ids = Vec::new();
        let mut last_pos = 0;
        
        // fancy-regex find_iter returns Result, so we need to handle it
        while last_pos < text.len() {
            match regex.find_from_pos(text, last_pos) {
                Ok(Some(mat)) => {
                    let chunk = mat.as_str();
                    let chunk_bytes = chunk.as_bytes();
                    let chunk_ids = self.encode_single_chunk(chunk_bytes);
                    ids.extend(chunk_ids);
                    last_pos = mat.end();
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        ids
    }

    /// Encode text into token IDs using BPE.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `allowed_special_tokens` - Policy for handling special tokens:
    ///     - None: Treat special tokens as regular text (encode with BPE)
    ///     - All: Encode special tokens as single token IDs
    ///     - NoneRaise: Raise error if special tokens found in text
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Example
    /// ```no_run
    /// use bpe_encode::{BpeEncode, AllowedSpecialTokens};
    /// 
    /// let mut encoder = BpeEncode::new();
    /// encoder.load("model.model").unwrap();
    /// let ids = encoder.encode("Hello world!", AllowedSpecialTokens::NoneRaise).unwrap();
    /// println!("{:?}", ids);
    /// ```
    pub fn encode(
        &self,
        text: &str,
        allowed_special_tokens: AllowedSpecialTokens,
    ) -> Result<Vec<u32>, String> {
        // Handle special tokens based on policy
        let allowed_special = match allowed_special_tokens {
            AllowedSpecialTokens::None => HashMap::new(),
            AllowedSpecialTokens::All => self.special_tokens.clone(),
            AllowedSpecialTokens::NoneRaise => {
                // Check if any special tokens are present
                for token in self.special_tokens.keys() {
                    if text.contains(token) {
                        return Err(format!(
                            "Special token '{}' found in text. Use allowed_special_tokens='all' to encode it.",
                            token
                        ));
                    }
                }
                HashMap::new()
            }
        };

        // If no special tokens to handle, use simple encoding
        if allowed_special.is_empty() {
            return Ok(self.encode_no_special_tokens(text));
        }

        // Split text by special tokens
        let special_pattern = format!(
            "({})",
            allowed_special
                .keys()
                .map(|k| fancy_regex::escape(k))
                .collect::<Vec<_>>()
                .join("|")
        );
        let special_regex = Regex::new(&special_pattern)
            .map_err(|e| format!("Failed to compile special token regex: {}", e))?;

        let mut ids = Vec::new();
        let mut last_end = 0;
        let mut pos = 0;

        // fancy-regex returns Result for find operations
        while pos < text.len() {
            match special_regex.find_from_pos(text, pos) {
                Ok(Some(mat)) => {
                    // Encode text before the special token
                    if mat.start() > last_end {
                        let before = &text[last_end..mat.start()];
                        ids.extend(self.encode_no_special_tokens(before));
                    }

                    // Add the special token
                    let special_token = mat.as_str();
                    if let Some(&token_id) = allowed_special.get(special_token) {
                        ids.push(token_id);
                    }

                    last_end = mat.end();
                    pos = mat.end();
                }
                Ok(None) => break,
                Err(e) => return Err(format!("Regex matching error: {}", e)),
            }
        }

        // Encode remaining text after last special token
        if last_end < text.len() {
            let remaining = &text[last_end..];
            ids.extend(self.encode_no_special_tokens(remaining));
        }

        Ok(ids)
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    /// * `ids` - Slice of token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Note
    /// Uses lossy UTF-8 conversion, replacing invalid sequences
    /// with the Unicode replacement character (�).
    ///
    /// # Example
    /// ```no_run
    /// use bpe_encode::BpeEncode;
    /// 
    /// let mut encoder = BpeEncode::new();
    /// encoder.load("model.model").unwrap();
    /// let text = encoder.decode(&[72, 101, 108, 108, 111]).unwrap();
    /// println!("{}", text); // "Hello"
    /// ```
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut byte_chunks = Vec::new();

        for &idx in ids {
            if let Some(bytes) = self.vocab.get(&idx) {
                // Regular token
                byte_chunks.extend_from_slice(bytes);
            } else if let Some(special) = self.inv_special_tokens.get(&idx) {
                // Special token
                byte_chunks.extend_from_slice(special.as_bytes());
            } else {
                return Err(format!("Invalid token ID: {}", idx));
            }
        }

        // Convert bytes to string with lossy conversion
        Ok(String::from_utf8_lossy(&byte_chunks).to_string())
    }
}

impl Default for BpeEncode {
    fn default() -> Self {
        Self::new()
    }
}


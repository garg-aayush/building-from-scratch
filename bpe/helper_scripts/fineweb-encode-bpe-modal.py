"""
Re-encode GPT-2 tokenized shards using custom BPE tokenizer

This script:
1. Reads existing GPT-2 tokenized shards (uint16 .npy files) from modal storage
2. Decodes them to text using tiktoken GPT-2 decoder
3. Re-encodes using the custom BPE tokenizer
4. Saves as new shards with 100M tokens per shard (per new tokenizer)
"""

import modal

# Create Modal app and image
app = modal.App("fineweb-bpe-reencode")

# Define the image with required dependencies
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("regex",
            "tiktoken",
            "numpy",
            "tqdm"
        ).add_local_file("../bpe.py", "/root/bpe.py")
        )

# Create a volume for persistent storage
volume = modal.Volume.from_name("fineweb-edu-data", create_if_missing=True)

# Configuration
SHARD_SIZE = int(1e8)  # 100M tokens per shard (for new tokenizer)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,
    retries=3,
)
def list_shards(max_train_shards: int = None):
    """List all shard files from the shards directory, separated by train/val"""
    import os
    
    shards_dir = "/data/shards"
    
    # Get all .npy files
    all_files = [f for f in os.listdir(shards_dir) if f.endswith(".npy")]
    
    # Separate train and val files
    train_files = [f for f in all_files if "train" in f]
    val_files = [f for f in all_files if "val" in f]
    
    train_files.sort()
    train_files = train_files[:max_train_shards]
    val_files.sort()
    
    print(f"Found {len(train_files)} training shard files")
    print(f"Found {len(val_files)} validation shard files")
    
    return {"train": train_files, "val": val_files}


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,  # 30 minutes timeout for larger shards
    cpu=4,
    memory=1024 * 16,
    retries=2,
)
def reencode_shard(shard_filename: str, vocab_path: str, suffix: str = "bpe") -> dict:
    """
    Re-encode a single shard from GPT-2 tokens to BPE tokens.
    
    Args:
        shard_filename: Name of the shard file (e.g., "finewebedu10T_train_0001.npy")
        vocab_path: Path to the BPE tokenizer model file (relative to /data/)
        suffix: Suffix for the tokenizer (default: "bpe")
    Returns:
        Dictionary with metadata about the processed shard
    """
    import os

    import numpy as np
    import tiktoken

    from bpe import BpeTokenizer
    
    shards_dir = "/data/shards"
    tokens_dir = f"/data/tokens_{suffix}"
    os.makedirs(tokens_dir, exist_ok=True)
    
    shard_path = f"{shards_dir}/{shard_filename}"
    tokens_path = f"{tokens_dir}/{shard_filename}"
    
    print(f"Processing shard: {shard_filename}")
    
    # Check if already processed
    if os.path.exists(tokens_path):
        print(f"Shard {shard_filename} already processed, skipping")
        return {
            "shard_filename": shard_filename,
            "status": "skipped",
            "tokens_path": tokens_path
        }
    
    # Load GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    gpt2_enc = tiktoken.get_encoding("gpt2")
    # set special tokens
    gpt2_enc._special_tokens["<|endoftext|>"] = 50256
    
    # Load BPE tokenizer
    print(f"Loading BPE tokenizer from {vocab_path}...")
    bpe_tokenizer = BpeTokenizer()
    full_vocab_path = "/data/" + vocab_path
    bpe_tokenizer.load(full_vocab_path)
    
    # Read GPT-2 tokenized shard
    print(f"Reading GPT-2 tokens from {shard_path}...")
    gpt2_tokens = np.load(shard_path)
    print(f"Loaded {len(gpt2_tokens):,} GPT-2 tokens")
    
    # Decode GPT-2 tokens to text
    print("Decoding GPT-2 tokens to text...")
    text = gpt2_enc.decode(gpt2_tokens.tolist())
    print(f"Decoded to text of length {len(text):,} characters")
    
    # Encode with BPE tokenizer
    print("Encoding with BPE tokenizer...")
    bpe_tokens = bpe_tokenizer.encode(text, allowed_special_tokens="all")
    print(f"Encoded to {len(bpe_tokens):,} BPE tokens")
    
    # Convert to numpy array (uint16)
    tokens_np = np.array(bpe_tokens, dtype=np.uint16)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), \
        "Token dictionary too large for uint16"
    
    # Save to the volume
    print(f"Saving BPE tokens to {tokens_path}...")
    np.save(tokens_path, tokens_np)
    print(f"Saved {len(tokens_np):,} tokens")
    
    volume.commit()
    
    return {
        "shard_filename": shard_filename,
        "status": "processed",
        "gpt2_tokens": len(gpt2_tokens),
        "bpe_tokens": len(bpe_tokens),
        "compression_ratio": len(gpt2_tokens) / len(bpe_tokens) if len(bpe_tokens) > 0 else 0,
        "tokens_path": tokens_path
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,
    cpu=4,
    memory=1024 * 16,
    retries=2,
)
def create_val_shard(suffix: str = "bpe"):
    """
    Create a single validation shard from re-encoded BPE tokens.
    Unlike training shards, validation is kept as a single shard regardless of size.
    """
    import os

    import numpy as np
    
    tokens_dir = f"/data/tokens_{suffix}"
    shards_new_dir = f"/data/shards_{suffix}"
    os.makedirs(shards_new_dir, exist_ok=True)
    
    # Get all re-encoded validation token files
    token_files = [f for f in os.listdir(tokens_dir) 
                   if f.endswith(".npy") and "val" in f]
    token_files.sort()
    print(f"Found {len(token_files)} re-encoded validation token files")
    
    if not token_files:
        print("No validation token files found")
        return
    
    # Concatenate all validation tokens into a single shard
    all_val_tokens = []
    for token_file in token_files:
        tokens_np = np.load(f"{tokens_dir}/{token_file}")
        print(f"Read {token_file}: {tokens_np.shape[0]:,} tokens")
        all_val_tokens.append(tokens_np)
    
    # Combine all validation tokens
    val_tokens_np = np.concatenate(all_val_tokens)
    print(f"Total validation tokens: {val_tokens_np.shape[0]:,}")
    
    # Save as a single validation shard
    save_path = f"{shards_new_dir}/finewebedu10T_bpe_val_0000.npy"
    print(f"Saving validation shard: {save_path}")
    np.save(save_path, val_tokens_np)
    
    volume.commit()
    print(f"Created single validation shard with {val_tokens_np.shape[0]:,} tokens")


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,
    cpu=4,
    memory=1024 * 16,
    retries=2,
)
def create_train_shards(suffix: str = "bpe"):
    """
    Create new training shards from re-encoded BPE tokens.
    Combines all re-encoded training tokens and splits into 100M token shards.
    """
    import os

    import numpy as np
    
    tokens_dir = f"/data/tokens_{suffix}"
    shards_new_dir = f"/data/shards_{suffix}"
    os.makedirs(shards_new_dir, exist_ok=True)
    
    # Get all re-encoded training token files
    token_files = [f for f in os.listdir(tokens_dir) 
                   if f.endswith(".npy") and "train" in f]
    token_files.sort()
    print(f"Found {len(token_files)} re-encoded training token files")
    
    if not token_files:
        print("No training token files found to create shards")
        return
    
    leftover_tokens_np = None
    shard_save_idx = 0
    
    for token_file in token_files:
        # Read the token file
        tokens_np = np.load(f"{tokens_dir}/{token_file}")
        print(f"Read {token_file}: {tokens_np.shape[0]:,} tokens")
        
        # Concatenate with leftover tokens from previous file
        if leftover_tokens_np is not None:
            tokens_np = np.concatenate([leftover_tokens_np, tokens_np])
            print(f"After concatenating leftovers: {tokens_np.shape[0]:,} tokens")
        
        # Calculate number of complete shards
        num_shards = tokens_np.shape[0] // SHARD_SIZE
        
        # Create shards
        for shard_idx in range(num_shards):
            # Get current shard (100M tokens)
            cur_shard_tokens_np = tokens_np[
                shard_idx * SHARD_SIZE:(shard_idx + 1) * SHARD_SIZE
            ]
            
            # Save the shard
            save_path = f"{shards_new_dir}/finewebedu10T_bpe_train_{shard_save_idx:04d}.npy"
            print(f"Saving train shard {shard_save_idx}: {save_path}")
            np.save(save_path, cur_shard_tokens_np)
            
            shard_save_idx += 1
        
        # Store leftover tokens for next iteration
        leftover_tokens_np = tokens_np[num_shards * SHARD_SIZE:]
        print(f"Leftover tokens: {leftover_tokens_np.shape[0]:,}")
        
        if leftover_tokens_np.shape[0] == 0:
            leftover_tokens_np = None
    
    # Save any remaining leftover tokens as final shard
    if leftover_tokens_np is not None and leftover_tokens_np.shape[0] > 0:
        save_path = f"{shards_new_dir}/finewebedu10T_bpe_train_{shard_save_idx:04d}.npy"
        print(f"Saving final leftover train shard {shard_save_idx}: {save_path}")
        print(f"Final shard size: {leftover_tokens_np.shape[0]:,} tokens")
        np.save(save_path, leftover_tokens_np)
    
    volume.commit()
    print(f"\nCreated {shard_save_idx + 1} training shards in {shards_new_dir}")


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,
)
def process_all_shards(vocab_path: str, max_train_shards: int = None, suffix: str = "bpe"):
    """
    Main orchestration function that:
    1. Lists all train and val shards
    2. Re-encodes them in parallel using BPE tokenizer
    3. Creates validation shard (single shard with original size)
    4. Creates training shards (100M tokens each)
    """
    print(f"Starting re-encoding process with vocab: {vocab_path}")
    
    # List all shards (train and val)
    shard_dict = list_shards.remote(max_train_shards=max_train_shards)
    train_files = shard_dict["train"]
    val_files = shard_dict["val"]
    
    all_files = train_files + val_files
    print(f"\nProcessing {len(all_files)} total shards ({len(train_files)} train, {len(val_files)} val)...")
    
    # Re-encode all shards in parallel
    results = list(reencode_shard.map(
        all_files,
        kwargs={"vocab_path": vocab_path, "suffix": suffix}
    ))
    
    # Print summary
    print("\n" + "="*60)
    print("Re-encoding Summary:")
    print("="*60)
    processed = sum(1 for r in results if r["status"] == "processed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    
    if processed > 0:
        total_gpt2 = sum(r.get("gpt2_tokens", 0) for r in results if r["status"] == "processed")
        total_bpe = sum(r.get("bpe_tokens", 0) for r in results if r["status"] == "processed")
        avg_compression = total_gpt2 / total_bpe if total_bpe > 0 else 0
        print(f"Total GPT-2 tokens: {total_gpt2:,}")
        print(f"Total BPE tokens: {total_bpe:,}")
        print(f"Average compression ratio: {avg_compression:.2f}x")
    print("="*60)
    
    # Create new shards from re-encoded tokens
    print("\nCreating validation shard (single shard)...")
    create_val_shard.remote(suffix=suffix)
    
    print("\nCreating training shards (100M tokens each)...")
    create_train_shards.remote(suffix=suffix)
    
    print("\nRe-encoding complete!")
    return results


@app.local_entrypoint()
def main(vocab_path: str = None, max_train_shards: int = None, suffix: str = "bpe"):
    """
    Main entry point for the re-encoding process.
    
    Args:
        vocab_path: Path to BPE tokenizer model file (relative to /data/ in modal volume)
                   Default: vocabs/tinystoriesv2-gpt4-train-16384.model
        max_train_shards: Maximum number of training shards to process (note: each shard is 100M tokens (GPT-2 tokens))
        suffix: Suffix for the tokenizer (default: "bpe")
    
    Usage:
        modal run fineweb-encode-bpe.py --vocab-path vocabs/tinystoriesv2-gpt4-train-16384.model --suffix tiny_16k --max-train-shards 25
    """
    assert max_train_shards is not None, "max_train_shards is required"
    print(f"Using BPE tokenizer: {vocab_path}, max_train_shards: {max_train_shards}, suffix: {suffix}")
    results = process_all_shards.remote(vocab_path, max_train_shards, suffix)
    print(f"\nProcessed {len(results)} shards successfully!")

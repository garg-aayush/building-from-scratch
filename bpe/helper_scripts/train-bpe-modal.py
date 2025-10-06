"""
Modal-based BPE Tokenizer Training

This script trains a Byte Pair Encoding (BPE) tokenizer using Modal compute
infrastructure (useful for training on large datasets)

What this does:
- Downloads training data from a provided HTTP URL to a persistent Modal volume (if not already present)
- Trains a BPE tokenizer on the data using parallel processing
- Saves the trained model and vocabulary files back to the Modal volume
- Allows you to download the trained artifacts locally

Input Requirements:
- Training data must be a SINGLE TEXT FILE (.txt)
- Either:
  1. Ensure the training data already exists at the given path in the Modal volume, OR
  2. Provide an HTTP URL (data_url) from which the script will download it automatically
- Make sure you have set the correct timeout/cpu/memory in the @app.function decorator
  as per your dataset size requirements

Output:
- After training completes, the model (.model) and vocabulary (.vocab) files are saved
  to the specified path on the Modal volume
- You can download these files locally using Modal's volume get command:
    modal volume get bpe-training-data <remote-file> <local-file>

Example workflow:
    # Train tokenizer (will auto-download data if not present)
    modal run train_bpe_modal.py \\
        --data-url "https://example.com/dataset.txt" \\
        --data-filename "dataset.txt" \\
        --vocab-size 2048 \\
        --output-prefix "dataset-2048"
    
    # Download trained model and vocab
    modal volume get bpe-training-data dataset-2048.model dataset-2048.model
    modal volume get bpe-training-data dataset-2048.vocab dataset-2048.vocab
"""

import modal

# Create Modal app
app = modal.App("bpe-tokenizer-trainer")

# Define the image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential", "wget")
        .pip_install("regex")  # Required for BPE tokenizer regex patterns
        .add_local_file("../bpe.py", "/root/bpe.py")
)

# Create a volume for persistent storage
volume = modal.Volume.from_name("bpe-training-data", create_if_missing=True)

@app.function(image=image, volumes={"/data": volume}, timeout=60 * 20)
def wget_to_volume(url: str, data_filename: str) -> str:
    """
    Download 'url' to the mounted Volume under /data/<data_filename>.
    Returns the absolute path written.
    """
    import subprocess
    from pathlib import Path

    root = Path("/data")
    dest = (root / data_filename)
    
    if dest.exists():
        print(f"File {data_filename} already exists: {dest}")
        return str(dest)
    
    # check is a path with directories  multi level
    if "/" in data_filename:
        dest.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "wget",
        "--no-verbose",
        "--timeout=30",
        "--tries=3",
        "-O", str(dest),
        url,
    ]
    print("Running cmd: ", cmd)
    subprocess.run(cmd, check=True)
        
    # make changes visible to other functions right away
    volume.commit()

    return str(dest)



@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,  # 1 hour timeout
    cpu=8,  # cpu * 2 vCPUs
    memory=1024 * 16,  # 128GB memory
    retries=2,
)
def train_bpe_tokenizer(
    vocab_size: int = 2048,
    input_filename: str = None,
    output_prefix: str = None,
    special_tokens: list[str] = None,
    boundary_split_token: str = None,
    num_processes: int = None,
):
    """
    Train BPE tokenizer on Modal.
    
    Args:
        vocab_size: Desired vocabulary size
        input_filename: Name of the input file in /data directory
        output_prefix: Prefix for saving model files (default: based on input filename and vocab size)
        special_tokens: List of special tokens (default: ["<|endoftext|>"])
        boundary_split_token: Token to split text into chunks (default: first special token)
        num_processes: Number of processes for parallel training (default: use all available CPUs)
    
    Returns:
        Dictionary with training results and saved file paths
    """
    import os
    import time

    from bpe import BpeTokenizer

    # Set defaults
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    if boundary_split_token is None:
        boundary_split_token = special_tokens[0]
    if num_processes is None:
        num_processes = os.cpu_count()
    if output_prefix is None:
        # Extract base name without extension
        base_name = input_filename.replace('.txt', '')
        output_prefix = f"{base_name}-{vocab_size}"
    
    # Paths
    input_file_path = "/data/" + input_filename
    output_file_prefix = "/data/" + output_prefix
    
    # Verify input file exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(
            f"Input file {input_file_path} not found on remote volume. "
            f"Please upload it first using the upload action."
        )
    
    print(f"Input file {input_file_path} found on remote volume")
    
    print("Training BPE tokenizer with the following configuration:")
    print(f"  Input file: {input_file_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Boundary split token: {boundary_split_token}")
    print(f"  Number of processes: {num_processes}")
    print(f"  Output prefix: {output_file_prefix}")
    
    # Get file size for reference
    file_size_mb = os.path.getsize(input_file_path) / (1024 * 1024)
    print(f"  Input file size: {file_size_mb:.2f} MB")
    
    # Initialize tokenizer
    start_time = time.time()
    tokenizer = BpeTokenizer()
    
    # Train the tokenizer
    print("\nStarting training...")
    tokenizer.train(
        input_file_path=input_file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        boundary_split_token=boundary_split_token,
        num_processes=num_processes,
        verbose=True
    )
    
    # Register special tokens
    # Assign special token IDs starting from vocab_size
    special_tokens_dict = {token: vocab_size + i for i, token in enumerate(special_tokens)}
    tokenizer.register_special_tokens(special_tokens_dict)
    
    # Save the tokenizer
    print(f"\nSaving tokenizer to {output_file_prefix}...")
    tokenizer.save(output_file_prefix)
    
    # Commit changes to volume
    volume.commit()
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Saved model to: {output_file_prefix}.model")
    print(f"Saved vocabulary to: {output_file_prefix}.vocab")
    
    return {
        "status": "success",
        "vocab_size": vocab_size,
        "model_file": f"{output_file_prefix}.model",
        "vocab_file": f"{output_file_prefix}.vocab",
        "training_time_seconds": training_time,
        "input_file_size_mb": file_size_mb,
    }


@app.local_entrypoint()
def main(
    data_url: str = None,
    data_filename: str = None,
    vocab_size: int = 2048,
    output_prefix: str = None,
):
    """
    Main entrypoint for BPE tokenizer training on Modal.
    """
    assert data_url is not None, "data_url is required"
    assert data_filename is not None, "data_filename is required"
    assert vocab_size is not None, "vocab_size is required"
    assert output_prefix is not None, "output_prefix is required"
    
    print("Training BPE tokenizer with the following configuration:")
    print(f"  Data URL: {data_url}")
    print(f"  Data filename: {data_filename}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Output prefix: {output_prefix}")
    
    # train the tokenizer
    result = train_bpe_tokenizer.remote(
        vocab_size=vocab_size,
        input_filename=data_filename,
        output_prefix=output_prefix,
    )
    print(f"Training result: {result}")



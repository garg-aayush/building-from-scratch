"""
Modal-optimized FineWeb-Edu dataset tokenization
Parallelizes download and tokenization using Modal's distributed compute
"""

import modal

# Create Modal app and image
app = modal.App("fineweb-edu-tokenizer")

# Define the image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets",
    "tiktoken",
    "numpy",
    "tqdm",
    "huggingface_hub",
    "pandas",
    "pyarrow",
    "requests"
)

# Create a volume for persistent storage
volume = modal.Volume.from_name("fineweb-edu-data", create_if_missing=True)

# Configuration
REMOTE_NAME = "10BT"
SHARD_SIZE = int(1e8)  # 100M tokens per shard
BATCH_SIZE = 1000  # Number of documents to process per function call

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    retries=3,
)
def get_parquet_urls():
    """Get all parquet file URLs from the HuggingFace dataset"""
    from huggingface_hub import HfApi
    
    api = HfApi()
    repo_id = "HuggingFaceFW/fineweb-edu"
    
    # List all files in the dataset repository
    files = api.list_repo_files(repo_id, repo_type="dataset")
    # Filter for parquet files in the sample-10BT split
    parquet_files = [f for f in files if f.startswith(f"sample/{REMOTE_NAME}/") and f.endswith(".parquet")]
    # sort parquet_files by filename
    parquet_files.sort()
    print(f"Found {len(parquet_files)} parquet files to download")
    # Create direct download URLs
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    parquet_urls = [base_url + f for f in parquet_files]
    results = list(download_parquet_file.map(parquet_urls))
    create_token_shards.remote()
    return results


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    cpu=4,
    memory=1024 * 16,
    retries=3,
)
def download_parquet_file(parquet_url: str) -> dict:
    """
    Download a single parquet file and save it to the volume.
    
    Args:
        parquet_url: URL of the parquet file to download
    
    Returns:
        Dictionary with metadata about the downloaded file.
    """
    import os

    import numpy as np
    import pandas as pd
    import requests
    import tiktoken

    # Extract filename from URL
    filename = parquet_url.split("/")[-1]
    save_parquet_path = f"/data/parquet/{filename}"
    save_tokens_path = f"/data/tokens/{filename.replace('.parquet', '.npy')}"
    os.makedirs(os.path.dirname(save_parquet_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_tokens_path), exist_ok=True)
    
    #########################################################
    # Download the parquet file to /data/filename
    #########################################################
    # Check if file already exists
    if os.path.exists(save_parquet_path):
        print(f"File {parquet_url} already exists: {filename}")
    else:
        response = requests.get(parquet_url, stream=True, timeout=300)
        response.raise_for_status()
        # save the response to the volume
        with open(save_parquet_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved file {parquet_url} to {save_parquet_path}")

    #########################################################
    # Tokenize the parquet file
    #########################################################
    print(f"Tokenizing file {save_parquet_path}")
    if os.path.exists(save_tokens_path):
        print(f"File {save_tokens_path} already exists: {filename}")
    else:
        # read the parquet file
        df = pd.read_parquet(save_parquet_path)
        text_column = df['text']
        
        # tokenize the text column
        tokens_list = []
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens["<|endoftext|>"]  # end of text token
        for text in text_column:
            tokens = [eot]  # the special <|endoftext|> token delimits all documents
            tokens.extend(enc.encode_ordinary(text))
            tokens_list.extend(tokens)
        
        # convert to numpy array (uint16)
        tokens_np = np.array(tokens_list)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np = tokens_np.astype(np.uint16)
        
        # save to the volume
        print(f"{save_tokens_path}: tokens_np.dtype: {tokens_np.dtype}, tokens_np.shape: {tokens_np.shape}")
        np.save(save_tokens_path, tokens_np)
        print(f"Saved tokens to {save_tokens_path}")

    volume.commit()
    
    return {
        "filename": filename,
        "parquet_url": parquet_url,
        "save_parquet_path": save_parquet_path,
        "save_tokens_path": save_tokens_path
    }

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    cpu=4,
    memory=1024 * 16,
    retries=3,
)
def create_token_shards():
    """Create token shards from the tokenized files"""
    import os

    import numpy as np
    
    shard_size = int(1e8)  # 100M tokens per shard
    
    # get all token files
    token_files = [f for f in os.listdir("/data/tokens") if f.endswith(".npy")]
    token_files.sort()
    print(f"Found {len(token_files)} token files")
    
    # shards directory
    tokens_dir = "/data/tokens"
    shards_dir = "/data/shards"
    os.makedirs(shards_dir, exist_ok=True)
    
    leftover_tokens_np = None
    shard_save_idx = 0
    for token_file in token_files:
        # read the token file
        tokens_np = np.load(f"{tokens_dir}/{token_file}")
        print(f"Read {token_file}: tokens_np.shape: {tokens_np.shape}")
        
        # concatenate the leftover tokens
        if leftover_tokens_np is not None:
            tokens_np = np.concatenate([leftover_tokens_np, tokens_np])
        print(f"After concatenating {token_file}: tokens_np.shape: {tokens_np.shape}")
        
        # total number of shards from the token file
        num_shards = tokens_np.shape[0] // shard_size
        for shard_idx in range(num_shards):
            # set the split
            split = "val" if shard_save_idx == 0 else "train"
            
            # get the current shard
            cur_shard_tokens_np = tokens_np[shard_idx * shard_size:(shard_idx + 1) * shard_size]
            
            # save the shard
            save_path = f"{shards_dir}/finewebedu10T_{split}_{shard_save_idx:04d}.npy"
            print(f"Saving shard {shard_save_idx} to {save_path}")
            np.save(save_path, cur_shard_tokens_np)
            
            # update the shard save index
            shard_save_idx += 1
        
        # leftover tokens
        leftover_tokens_np = tokens_np[num_shards * shard_size:]
        print(f"leftover_tokens_np.shape: {leftover_tokens_np.shape}")
        if leftover_tokens_np.shape[0] == 0: leftover_tokens_np = None
    
    # save the leftover tokens
    if leftover_tokens_np is not None:
        save_path = f"{shards_dir}/finewebedu10T_{split}_{shard_save_idx:04d}.npy"
        print(f"Saving leftover tokens: {leftover_tokens_np.shape} to {save_path}")
        np.save(save_path, leftover_tokens_np)
        
    volume.commit()

@app.local_entrypoint()
def main():
    get_parquet_urls.remote()
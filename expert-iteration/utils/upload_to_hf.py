#!/usr/bin/env python3
"""
Simple script to upload checkpoints to Hugging Face Hub.
"""
import os
from huggingface_hub import HfApi, create_repo

# Configuration
RESULTS_DIR = "/root/RESULTS/"
REPO_NAME = "cs336_exp-iter_exps"  # Change this to your desired repo name
PRIVATE = False  # Set to True if you want a private repo

# which directory to upload
CHECKPOINTS = os.listdir(RESULTS_DIR)

# Upload evaluation results?
UPLOAD_EVAL = False


def main():
    api = HfApi()
    
    # Get username
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/{REPO_NAME}"
    
    print(f"Creating repository: {repo_id}")
    create_repo(repo_id, repo_type="model", private=PRIVATE, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{repo_id}\n")
    
    # Upload checkpoints
    for checkpoint in CHECKPOINTS:
        checkpoint_path = f"{RESULTS_DIR}/{checkpoint}"
        print(f"Uploading {checkpoint}...")
        
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=checkpoint,
            commit_message=f"Upload {checkpoint}"
        )
        print(f"{checkpoint} uploaded\n")
    
    # Upload evaluation results
    if UPLOAD_EVAL:
        eval_path = f"{RESULTS_DIR}/eval_examples"
        print("Uploading evaluation results...")
        
        api.upload_folder(
            folder_path=eval_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="eval_examples",
            commit_message="Upload evaluation examples"
        )
        print("Evaluation results uploaded\n")
    
    print(f"Done! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

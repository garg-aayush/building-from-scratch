#!/usr/bin/env python3
"""Script to setup data and models volume in Modal.

Run from inside the grpo/ directory:

    # Download model weights (default: Qwen/Qwen2.5-Math-1.5B):
    modal run utils/setup_modal.py::setup_model
    modal run utils/setup_modal.py::setup_model --repo-id Qwen/Qwen2.5-Math-1.5B

    # Download a specific checkpoint subfolder from a HF repo:
    modal run utils/setup_modal.py::setup_model \
        --repo-id garg-aayush/qwen-2.5-math-sft-filtered-2epoch \
        --subfolder checkpoint_56 \
        --dest-name qwen-2.5-math-sft-checkpoint56

    # Upload local training data:
    modal run utils/setup_modal.py::upload_data
    modal run utils/setup_modal.py::upload_data --local-data-dir /home/aayush/DATA/GRPO
"""

import os
from pathlib import Path

import modal

APP_NAME_SETUP = "grpo-setup"
CONTAINER_DATA_DIR = "/data"
VOLUME_NAME_DATA = "grpo-data"

# -------------------------------------------------------------#
# Modal app and volume
# -------------------------------------------------------------#
app = modal.App(APP_NAME_SETUP)
data_volume = modal.Volume.from_name(VOLUME_NAME_DATA, create_if_missing=True)


# -------------------------------------------------------------#
# Image
# -------------------------------------------------------------#
_hf_image = (
    modal.Image.debian_slim()
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# -------------------------------------------------------------#
# Remote functions
# -------------------------------------------------------------#
@app.function(
    image=_hf_image,
    volumes={CONTAINER_DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def _download_model_remote(
    repo_id: str,
    revision: str | None,
    subfolder: str | None,
    dest_name: str | None,
):
    import shutil
    from huggingface_hub import snapshot_download

    model_name = dest_name or repo_id.split("/")[-1]
    dest = os.path.join(CONTAINER_DATA_DIR, "models", model_name)
    os.makedirs(dest, exist_ok=True)

    allow_patterns = [f"{subfolder}/**", f"{subfolder}/*"] if subfolder else None
    print(f"Downloading {repo_id}{f'/{subfolder}' if subfolder else ''} -> {dest} ...")

    # Always download directly to dest (on the volume) to avoid cross-device moves.
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        revision=revision,
        allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN"),
    )

    if subfolder:
        # Files land at dest/<subfolder>/; move them up to dest/
        src = os.path.join(dest, subfolder)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Subfolder '{subfolder}' not found in repo")
        for item in os.listdir(src):
            os.rename(os.path.join(src, item), os.path.join(dest, item))
        os.rmdir(src)
        print(f"  moved {subfolder}/ contents up to {dest}")

    data_volume.commit()
    print("Done.")


@app.function(
    image=modal.Image.debian_slim(),
    volumes={CONTAINER_DATA_DIR: data_volume},
    timeout=3600,
)
def _write_files_remote(file_data: list[tuple[str, bytes]]):
    for rel_path, content in file_data:
        dest = os.path.join(CONTAINER_DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Path(dest).write_bytes(content)
        print(f"  wrote {dest}")
    data_volume.commit()


# -------------------------------------------------------------#
# Local entrypoints
# -------------------------------------------------------------#
@app.local_entrypoint()
def setup_model(
    repo_id: str = "Qwen/Qwen2.5-Math-1.5B",
    revision: str = None,
    subfolder: str = None,
    dest_name: str = None,
):
    """Download a Hugging Face model into the grpo-data volume.

    Use --subfolder to pull only a specific checkpoint directory from the repo.
    Use --dest-name to give the downloaded model a custom name under /data/models/.
    """
    print(f"Downloading {repo_id}{f'/{subfolder}' if subfolder else ''} into grpo-data volume ...")
    _download_model_remote.remote(
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        dest_name=dest_name,
    )


@app.local_entrypoint()
def upload_data(local_data_dir: str = "/home/aayush/DATA/GRPO"):
    """Upload local data files into the grpo-data volume at /data/GRPO/."""
    local_path = Path(local_data_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Not found: {local_data_dir}")

    files: list[tuple[str, bytes]] = []
    for p in sorted(local_path.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_path)
            print(f"  reading {p}")
            files.append((f"GRPO/{rel}", p.read_bytes()))

    print(f"Uploading {len(files)} file(s) to grpo-data volume ...")
    _write_files_remote.remote(files)
    print("Upload complete.")

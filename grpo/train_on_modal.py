#!/usr/bin/env python3
"""Run GRPO training on Modal.

Prerequisites (one-time setup, run from inside grpo/):
    modal run setup_modal.py::setup_model
    modal run setup_modal.py::upload_data --local-data-dir /home/aayush/DATA/GRPO

Usage (run from inside grpo/):
    modal run train_on_modal.py --config configs/test_modal.yaml

    # Always set --output-dir to a path inside the results volume:
    modal run train_on_modal.py --config configs/test_modal.yaml --output-dir /results/test_modal

    # --detach is REQUIRED with --spawn; without it Modal kills the job on exit.
    modal run --detach train_on_modal.py --config configs/test_modal.yaml --output-dir /results/test_modal --spawn

Default paths inside Modal containers:
    Model weights : /data/models/Qwen2.5-Math-1.5B
    Train data    : /data/GRPO/train.jsonl
    Val data      : /data/GRPO/validation.jsonl
    Prompt tmpl   : /data/GRPO/r1_zero.prompt
    Outputs       : /results/grpo   <--- override with --output-dir per run
"""

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import modal
from omegaconf import OmegaConf

# -------------------------------------------------------------#
# Default paths (all inside Modal volumes)
# -------------------------------------------------------------#
# Volume names
VOLUME_NAME_DATA    = "grpo-data"
VOLUME_NAME_RESULTS = "grpo-results"

# Container mount points
CONTAINER_DATA_DIR    = "/data"
CONTAINER_RESULTS_DIR = "/results"
CONTAINER_CODE_DIR    = "/grpo"

# GPU
GPU_TYPE = "H100"

# Modal app names
APP_NAME_TRAINING = "grpo-training"

DEFAULT_MODEL_DIR  = f"{CONTAINER_DATA_DIR}/models/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA = f"{CONTAINER_DATA_DIR}/GRPO/train.jsonl"
DEFAULT_VAL_DATA   = f"{CONTAINER_DATA_DIR}/GRPO/validation.jsonl"
DEFAULT_PROMPT_TPL = f"{CONTAINER_DATA_DIR}/GRPO/r1_zero.prompt"
DEFAULT_OUTPUT_DIR = f"{CONTAINER_RESULTS_DIR}/grpo"


# -------------------------------------------------------------#
# Modal app and volumes
# -------------------------------------------------------------#
app = modal.App(APP_NAME_TRAINING)
data_volume    = modal.Volume.from_name(VOLUME_NAME_DATA,    create_if_missing=True)
results_volume = modal.Volume.from_name(VOLUME_NAME_RESULTS, create_if_missing=True)

# -------------------------------------------------------------#
# Container image
# -------------------------------------------------------------#
_GRPO_ROOT = Path(__file__).parent

_training_image = (
    # CUDA 12.8: closest stable image to the cu129 build used locally.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "vllm==0.11.0",
        "transformers==4.57.1",
        "bitsandbytes==0.49.2",
        "omegaconf",
        "wandb",
        "tqdm",
        "datasets",
        "huggingface_hub[hf_transfer]",
        "sentencepiece",
        "latex2sympy2_extended[antlr4_13_2]",
        "math-verify[antlr4_13_2]",
        "pylatexenc",
        "accelerate",
        "wheel",
        "packaging",
        "setuptools",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .uv_pip_install(
        "flash-attn==2.8.3",
        extra_options="--no-build-isolation",
    )
    # Bake grpo source files into the image.
    .add_local_file(
        str(_GRPO_ROOT / "train_grpo.py"),
        remote_path=f"{CONTAINER_CODE_DIR}/train_grpo.py",
        copy=True,
    )
    .add_local_dir(
        str(_GRPO_ROOT / "utils"),
        remote_path=f"{CONTAINER_CODE_DIR}/utils",
        copy=True,
    )
    .add_local_dir(
        str(_GRPO_ROOT / "configs"),
        remote_path=f"{CONTAINER_CODE_DIR}/configs",
        copy=True,
    )
)

# -------------------------------------------------------------#
# Training function
# -------------------------------------------------------------#
@app.function(
    image=_training_image,
    gpu=GPU_TYPE,
    volumes={
        CONTAINER_DATA_DIR:    data_volume,
        CONTAINER_RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),       # WANDB_API_KEY
        modal.Secret.from_name("huggingface-secret"), # HF_TOKEN (optional)
    ],
    timeout=int(24 * 3600),
)
def run_training(merged_config_yaml: str):
    """Execute train_grpo.py with a fully-merged OmegaConf config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(merged_config_yaml)
        tmp_config = f.name

    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env["PYTHONPATH"] = CONTAINER_CODE_DIR

    result = subprocess.run(
        [sys.executable, f"{CONTAINER_CODE_DIR}/train_grpo.py", "--config", tmp_config],
        env=env,
    )
    # Exit code -11 (SIGSEGV) is a known benign crash from vLLM/NCCL during
    # process cleanup after training completes. Treat it as success.
    if result.returncode not in (0, -11):
        raise RuntimeError(f"train_grpo.py exited with code {result.returncode}")


# -------------------------------------------------------------#
# Local entrypoint
# -------------------------------------------------------------#
@app.local_entrypoint()
def main(
    config: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    train_data: str = DEFAULT_TRAIN_DATA,
    val_data: str = DEFAULT_VAL_DATA,
    prompt_template: str = DEFAULT_PROMPT_TPL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    spawn: bool = False,
):
    base_cfg = OmegaConf.load(config)
    path_overrides = OmegaConf.create({
        "paths": {
            "model_path":           model_dir,
            "train_data_file":      train_data,
            "val_data_file":        val_data,
            "prompt_template_file": prompt_template,
            "output_dir":           output_dir,
        }
    })
    merged = OmegaConf.merge(base_cfg, path_overrides)

    buf = io.StringIO()
    OmegaConf.save(merged, buf)

    if spawn:
        call = run_training.spawn(merged_config_yaml=buf.getvalue())
        print("Job submitted!")
        print(f"  call ID : {call.object_id}")
        print(f"  config  : {config}")
        print(f"  output  : {output_dir}")
        print(f"  track   : https://modal.com/apps/{APP_NAME_TRAINING}")
        print()
        print("NOTE: run with 'modal run --detach' to prevent the app from")
        print("      being torn down when the local entrypoint exits.")
    else:
        print("Submitting GRPO training job")
        print(f"  config : {config}")
        print(f"  output : {output_dir}")
        run_training.remote(merged_config_yaml=buf.getvalue())

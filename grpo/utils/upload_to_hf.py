#!/usr/bin/env python3
"""
Upload GRPO experiment checkpoints and rollouts from Modal grpo-results volume
to a public HuggingFace Hub repository.

Upload policy:
  crashed  → last available checkpoint  + all rollout jsonls
  finished → last + best checkpoints    + all rollout jsonls
  (if best == last, only one checkpoint folder is uploaded)

HF repo layout:
  {run_name}/
    checkpoint_{step:03d}/        ← model / tokenizer files
    rollouts/
      rollouts_step_{step:03d}.jsonl

A metadata.json is written inside each run's folder summarising last / best steps.

Usage (run from inside grpo/):
    modal run utils/upload_to_hf.py --hf-repo <owner/cs336-grpo-exps>
    modal run utils/upload_to_hf.py --hf-repo <owner/cs336-grpo-exps> --dry-run
"""

import json
import os
from pathlib import Path

import modal
import pandas as pd

# ------------------------------------------------------------------ #
# Modal plumbing
# ------------------------------------------------------------------ #
VOLUME_NAME_RESULTS = "grpo-results"
CONTAINER_RESULTS_DIR = "/results"
APP_NAME = "grpo-upload-to-hf"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME_RESULTS, create_if_missing=False)

_upload_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]", "pandas")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ------------------------------------------------------------------ #
# run_name → output_dir mapping (derived from configs/)
# ------------------------------------------------------------------ #
# fmt: off
RUN_OUTPUT_DIRS: dict[str, str] = {
    "len_norm_constant":                  "/results/length_normalization/len_norm_constant",
    "len_norm_mean":                       "/results/length_normalization/len_norm_mean",
    "len_norm_microbatch":                 "/results/length_normalization/len_norm_microbatch",
    "lr_sweep_1e-4":                       "/results/lr_sweep/lr_1e-4.yaml",
    "lr_sweep_1e-5":                       "/results/lr_sweep/lr_1e-5.yaml",
    "lr_sweep_1e-6":                       "/results/lr_sweep/lr_1e-6.yaml",
    "lr_sweep_1.5e-5":                     "/results/lr_sweep/lr_1.5e-5.yaml",
    "lr_sweep_3e-5":                       "/results/lr_sweep/lr_3e-5.yaml",
    "lr_sweep_3e-6":                       "/results/lr_sweep/lr_3e-6.yaml",
    "no_baseline":                         "/results/baselines/no_baseline",
    "no_baseline_lr1.5e-5":               "/results/baselines/no_baseline_lr1.5e-5",
    "reinforce_baseline":                  "/results/baselines/reinforce_baseline",
    "off_policy_e1_tb128_ga32":           "/results/off_policy_sweep/e1_tb128_ga32",
    "off_policy_e1_tb256_ga64":           "/results/off_policy_sweep/e1_tb256_ga64",
    "off_policy_e2_tb128_ga32":           "/results/off_policy_sweep/e2_tb128_ga32",
    "off_policy_e2_tb256_ga64":           "/results/off_policy_sweep/e2_tb256_ga64",
    "off_policy_e4_tb256_ga64":           "/results/off_policy_sweep/e4_tb256_ga64",
    "off_policy_e4_tb64_ga16":            "/results/off_policy_sweep/e4_tb64_ga16",
    "off_policy_full_e1_tb128_ga32":      "/results/off_policy_sweep/full_e1_tb128_ga32",
    "off_policy_full_e1_tb256_ga64":      "/results/off_policy_sweep/full_e1_tb256_ga64",
    "off_policy_full_e2_tb256_ga64":      "/results/off_policy_sweep/full_e2_tb256_ga64",
    "off_policy_no_clip_e1_tb128_ga32":   "/results/off_policy_sweep/no_clip_e1_tb128_ga32",
    "off_policy_no_clip_e1_tb256_ga64":   "/results/off_policy_sweep/no_clip_e1_tb256_ga64",
    "question_only_e1_tb256_ga64":        "/results/question_only/e1_tb256_ga64",
    "std_dev":                             "/results/std_dev/std_dev",
    "sft_grpo_e1_tb256_ga64":            "/results/sft_grpo/e1_tb256_ga64",
    "sft_grpo_e1_tb256_ga64_ckpt10":     "/results/sft_grpo/e1_tb256_ga64_ckpt10",
    "sft_grpo_e1_tb256_ga64_ckpt20":     "/results/sft_grpo/e1_tb256_ga64_ckpt20",
    "sft_grpo_e1_tb256_ga64_lr1.5e-5":   "/results/sft_grpo/e1_tb256_ga64_lr1.5e-5",
}
# fmt: on


# ------------------------------------------------------------------ #
# Modal function: runs inside container with the results volume mounted
# ------------------------------------------------------------------ #
@app.function(
    image=_upload_image,
    volumes={CONTAINER_RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN
    timeout=int(6 * 3600),
    memory=4096,
)
def upload_run(
    run_name: str,
    output_dir: str,
    state: str,
    last_grpo_step: int,
    best_eval_grpo_step: int | None,
    hf_repo: str,
    dry_run: bool,
) -> dict:
    """Upload one experiment's checkpoints and rollouts to HF Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    # Create repo if it doesn't exist yet (idempotent)
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True, private=False)
    base_path = Path(output_dir)
    results: dict = {"run_name": run_name, "uploaded": [], "skipped": [], "errors": []}

    if not base_path.exists():
        results["errors"].append(f"output_dir not found: {output_dir}")
        return results

    # ---- find available checkpoints --------------------------------
    available_ckpts: dict[int, Path] = {}
    for d in sorted(base_path.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint_"):
            try:
                step = int(d.name.split("_")[1])
                available_ckpts[step] = d
            except (IndexError, ValueError):
                pass

    if not available_ckpts:
        results["errors"].append("no checkpoints found")
        return results

    # ---- decide which checkpoints to upload ------------------------
    # "last" = highest step available (may differ from last_grpo_step if
    # the run crashed before the next checkpoint_interval boundary)
    actual_last_step = max(available_ckpts)

    steps_to_upload: dict[str, int] = {"checkpoint_last": actual_last_step}
    if state == "finished" and best_eval_grpo_step is not None:
        # find the nearest checkpoint at or before best_eval_grpo_step
        candidates = [s for s in available_ckpts if s <= best_eval_grpo_step]
        if candidates:
            best_ckpt_step = max(candidates)
            if best_ckpt_step != actual_last_step:
                steps_to_upload["checkpoint_best"] = best_ckpt_step

    # ---- upload checkpoints ----------------------------------------
    for label, step in steps_to_upload.items():
        ckpt_path = available_ckpts[step]
        hf_folder = f"{run_name}/{label}"
        print(f"  [{run_name}] uploading {ckpt_path.name} → {hf_repo}/{hf_folder}")
        if not dry_run:
            api.upload_folder(
                folder_path=str(ckpt_path),
                repo_id=hf_repo,
                path_in_repo=hf_folder,
                repo_type="model",
            )
        results["uploaded"].append(f"{hf_folder} (step {step})")

    # ---- upload rollout jsonls (last + best steps only) ------------
    rollouts_dir = base_path / "rollouts"
    if rollouts_dir.exists():
        # Build step → file mapping
        available_rollouts: dict[int, Path] = {}
        for f in rollouts_dir.iterdir():
            if not f.name.endswith(".jsonl"):
                continue
            try:
                step = int(f.stem.split("_step_")[1])
                available_rollouts[step] = f
            except (IndexError, ValueError):
                pass

        # Pick rollout at or before each target step
        rollout_steps_to_upload: dict[str, int] = {}
        for label, target_step in steps_to_upload.items():
            candidates = [s for s in available_rollouts if s <= target_step]
            if candidates:
                rollout_steps_to_upload[label] = max(candidates)

        # Upload, deduplicating if last == best
        uploaded_rollout_steps: set[int] = set()
        for label, step in rollout_steps_to_upload.items():
            if step in uploaded_rollout_steps:
                continue
            uploaded_rollout_steps.add(step)
            rollout_file = available_rollouts[step]
            hf_path = f"{run_name}/rollouts/{rollout_file.name}"
            print(f"  [{run_name}] uploading rollout {rollout_file.name} ({label})")
            if not dry_run:
                api.upload_file(
                    path_or_fileobj=str(rollout_file),
                    path_in_repo=hf_path,
                    repo_id=hf_repo,
                    repo_type="model",
                )
            results["uploaded"].append(hf_path)
    else:
        results["skipped"].append("rollouts dir not found")

    # ---- write metadata.json ---------------------------------------
    metadata = {
        "run_name": run_name,
        "state": state,
        "last_grpo_step": last_grpo_step,
        "actual_last_ckpt_step": actual_last_step,
        "best_eval_grpo_step": best_eval_grpo_step,
        "best_ckpt_step": steps_to_upload.get("checkpoint_best", actual_last_step),
        "available_checkpoints": sorted(available_ckpts),
    }
    meta_bytes = json.dumps(metadata, indent=2).encode()
    hf_meta_path = f"{run_name}/metadata.json"
    print(f"  [{run_name}] writing {hf_meta_path}")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=meta_bytes,
            path_in_repo=hf_meta_path,
            repo_id=hf_repo,
            repo_type="model",
        )
    results["uploaded"].append(hf_meta_path)

    return results


# ------------------------------------------------------------------ #
# Local entrypoint
# ------------------------------------------------------------------ #
@app.local_entrypoint()
def main(
    hf_repo: str,
    csv_path: str = "play-scripts/grpo_run_summary.csv",
    dry_run: bool = False,
):
    """
    hf_repo   : HuggingFace repo id, e.g. "garg-aayush/cs336-grpo-exps"
    csv_path  : path to grpo_run_summary.csv (relative to grpo/ dir)
    dry_run   : if True, print what would be uploaded without actually uploading
    """
    df = pd.read_csv(csv_path)

    # Deduplicate: if the same run_name appears twice (one crashed, one finished)
    # keep the finished run; otherwise keep the one with the higher last_grpo_step.
    state_rank = {"finished": 1, "crashed": 0, "running": 0}
    df["_rank"] = df["state"].map(lambda s: state_rank.get(s, 0))
    df = (
        df.sort_values(["run_name", "_rank", "last_grpo_step"], ascending=[True, False, False])
        .drop_duplicates(subset=["run_name"], keep="first")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    # Build upload jobs
    jobs = []
    skipped = []
    for _, row in df.iterrows():
        run_name = row["run_name"]
        output_dir = RUN_OUTPUT_DIRS.get(run_name)
        if output_dir is None:
            skipped.append(f"{run_name}: no output_dir mapping — add to RUN_OUTPUT_DIRS")
            continue
        jobs.append(
            dict(
                run_name=run_name,
                output_dir=output_dir,
                state=row["state"],
                last_grpo_step=int(row["last_grpo_step"]),
                best_eval_grpo_step=(
                    int(row["best_eval_grpo_step"])
                    if pd.notna(row.get("best_eval_grpo_step"))
                    else None
                ),
                hf_repo=hf_repo,
                dry_run=dry_run,
            )
        )

    if skipped:
        print("\nSkipped (no output_dir mapping):")
        for s in skipped:
            print(f"  {s}")

    print(f"\nUploading {len(jobs)} runs to {hf_repo} (dry_run={dry_run})\n")
    if dry_run:
        print("[DRY RUN] — no files will actually be uploaded\n")

    # Run sequentially — parallel commits to the same HF repo cause 412
    # (optimistic concurrency: each commit must reference the current HEAD,
    # so concurrent writers constantly collide).
    all_results = []
    for j in jobs:
        print(f"→ {j['run_name']} ...")
        r = upload_run.remote(**j)
        all_results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    for r in all_results:
        if isinstance(r, Exception):
            print(f"\n[EXCEPTION] {r}")
            continue
        status = "OK" if not r["errors"] else "ERROR"
        print(f"\n[{status}] {r['run_name']}")
        for item in r["uploaded"]:
            print(f"      ✓  {item}")
        for item in r["skipped"]:
            print(f"      -  {item}")
        for err in r["errors"]:
            print(f"      ✗  {err}")

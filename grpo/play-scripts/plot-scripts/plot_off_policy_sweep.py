#!/usr/bin/env python3
"""
Plot broad off-policy sweep results for GRPO experiments, fetched from W&B.

Creates a single-row figure with 3 subplots:
1. Eval Reward
2. Gradient Norm
3. Mean Response Length

Runs vary epochs_per_rollout_batch and train_batch_size, which together determine
how many gradient update steps happen per GRPO step.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT = "garg-aayush/grpo"

# Ordered from on-policy to most off-policy (by opt steps per GRPO step)
RUN_NAMES = [
    "off_policy_e1_tb256_ga64",  # 1 opt step  (on-policy baseline)
    "off_policy_e1_tb128_ga32",  # 2 opt steps (smaller batch)
    "off_policy_e2_tb256_ga64",  # 2 opt steps (more epochs)
    "off_policy_e2_tb128_ga32",  # 4 opt steps
    "off_policy_e4_tb256_ga64",  # 4 opt steps (more epochs)
    "off_policy_e4_tb64_ga16",   # 16 opt steps (aggressive)
]

DISPLAY_NAMES = {
    "off_policy_e1_tb256_ga64": "epochs=1, train_batch=256 — on-policy (1 opt step)",
    "off_policy_e1_tb128_ga32": "epochs=1, train_batch=128 — mild off-policy (2 opt steps)",
    "off_policy_e2_tb256_ga64": "epochs=2, train_batch=256 — mild off-policy (2 opt steps)",
    "off_policy_e2_tb128_ga32": "epochs=2, train_batch=128 — moderate off-policy (4 opt steps)",
    "off_policy_e4_tb256_ga64": "epochs=4, train_batch=256 — moderate off-policy (4 opt steps)",
    "off_policy_e4_tb64_ga16":  "epochs=4, train_batch=64  — aggressive off-policy (16 opt steps)",
}

# Colorblind-friendly palette (6 colors) — first 3 match baseline ablation script
COLORS = {
    "off_policy_e1_tb256_ga64": "#E69F00",  # Orange
    "off_policy_e1_tb128_ga32": "#56B4E9",  # Sky blue
    "off_policy_e2_tb256_ga64": "#009E73",  # Bluish green
    "off_policy_e2_tb128_ga32": "#0072B2",  # Blue
    "off_policy_e4_tb256_ga64": "#CC79A7",  # Pink
    "off_policy_e4_tb64_ga16":  "#D55E00",  # Vermillion
}

METRICS = {
    "eval/reward": {
        "ylabel": "Reward Accuracy",
        "title": "Eval Reward Accuracy",
        "style": "o-",
        "markersize": 4,
        "linewidth": 2,
        "log_scale": False,
        "ylim_min": None,
        "ylim_max": None,
    },
"train/grad_norm": {
        "ylabel": "Gradient Norm",
        "title": "Gradient Norm",
        "style": "-",
        "markersize": 0,
        "linewidth": 1.5,
        "log_scale": False,
        "ylim_min": -0.01,
        "ylim_max": 2.0,
    },
    "train/mean_response_length": {
        "ylabel": "Mean Response Length (tokens)",
        "title": "Mean Response Length",
        "style": "-",
        "markersize": 0,
        "linewidth": 1.5,
        "log_scale": False,
        "ylim_min": None,
        "ylim_max": None,
    },
}


def fetch_wandb_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch metric histories from W&B for each run."""
    api = wandb.Api()
    runs = api.runs(PROJECT)

    all_data: dict[str, dict[str, pd.DataFrame]] = {}

    for run in runs:
        if run.name not in RUN_NAMES:
            continue
        print(f"Fetching run: {run.name} (id: {run.id})")

        run_metrics = {}
        for metric_key in METRICS:
            history = run.history(keys=[metric_key, "grpo_step"])
            if history.empty:
                print(f"  Warning: No data for {metric_key} in {run.name}")
                continue
            df = history.dropna(subset=[metric_key, "grpo_step"])
            df = df.rename(columns={"grpo_step": "step", metric_key: "value"})
            df = df[["step", "value"]].sort_values("step").reset_index(drop=True)
            run_metrics[metric_key] = df
            print(f"  {metric_key}: {len(df)} points")

        all_data[run.name] = run_metrics

    return all_data


def save_csv(all_data: dict[str, dict[str, pd.DataFrame]], output_dir: Path) -> None:
    """Save fetched data to CSV for reproducibility."""
    rows = []
    for run_name, metrics in all_data.items():
        for metric_key, df in metrics.items():
            for _, row in df.iterrows():
                rows.append({
                    "run_name": run_name,
                    "metric": metric_key,
                    "step": int(row["step"]),
                    "value": row["value"],
                })
    csv_path = output_dir / "off_policy_sweep_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


def plot_results(
    all_data: dict[str, dict[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    """Create a 1x4 figure with the four metric plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fig.suptitle(
        "Broad off-policy sweep: GRPO experiments",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, (metric_key, meta) in zip(axes, METRICS.items()):
        for run_name in RUN_NAMES:
            if run_name not in all_data or metric_key not in all_data[run_name]:
                continue
            df = all_data[run_name][metric_key]
            ax.plot(
                df["step"],
                df["value"],
                meta["style"],
                label=DISPLAY_NAMES[run_name],
                color=COLORS[run_name],
                linewidth=meta["linewidth"],
                markersize=meta["markersize"],
                alpha=0.9,
            )

        if meta["log_scale"]:
            ax.set_yscale("log")
        if meta["ylim_min"] is not None or meta["ylim_max"] is not None:
            ax.set_ylim(bottom=meta["ylim_min"], top=meta["ylim_max"])
        ax.set_xlabel("GRPO Step", fontsize=10)
        ax.set_ylabel(meta["ylabel"], fontsize=10)
        ax.set_title(meta["title"], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=9)

    # Single shared legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "results" / "off_policy_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching runs from {PROJECT}...")
    all_data = fetch_wandb_data()

    if not all_data:
        print("No runs found!")
        return

    save_csv(all_data, output_dir)
    plot_results(all_data, output_dir / "off_policy_sweep.png")


if __name__ == "__main__":
    main()

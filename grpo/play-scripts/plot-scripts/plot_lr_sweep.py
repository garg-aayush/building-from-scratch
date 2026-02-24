#!/usr/bin/env python3
"""
Plot learning rate sweep results for GRPO experiments, fetched from W&B.

Creates a single-row figure with 2 subplots:
1. Eval Reward
2. Train Entropy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT = "garg-aayush/grpo"
RUN_NAMES = [
    "lr_sweep_1e-4",
    "lr_sweep_3e-5",
    "lr_sweep_1.5e-5",
    "lr_sweep_1e-5",
    "lr_sweep_3e-6",
    "lr_sweep_1e-6",
]

DISPLAY_NAMES = {
    "lr_sweep_1e-4": "lr=1e-4",
    "lr_sweep_3e-5": "lr=3e-5",
    "lr_sweep_1.5e-5": "lr=1.5e-5",
    "lr_sweep_1e-5": "lr=1e-5",
    "lr_sweep_3e-6": "lr=3e-6",
    "lr_sweep_1e-6": "lr=1e-6",
}

COLORS = {
    "lr_sweep_1e-4": "#E69F00",    # Orange
    "lr_sweep_3e-5": "#56B4E9",    # Sky blue
    "lr_sweep_1.5e-5": "#009E73",  # Bluish green
    "lr_sweep_1e-5": "#CC79A7",    # Reddish purple
    "lr_sweep_3e-6": "#D55E00",    # Vermillion
    "lr_sweep_1e-6": "#0072B2",    # Blue
}

METRICS = {
    "eval/reward": {
        "ylabel": "Reward Accuracy",
        "title": "Eval Reward Accuracy",
        "style": "o-",
        "markersize": 4,
    },
    "train/entropy": {
        "ylabel": "Token Entropy",
        "title": "Token Entropy",
        "style": "-",
        "markersize": 0,
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
            df = history.dropna(subset=[metric_key])
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
    pd.DataFrame(rows).to_csv(output_dir / "lr_sweep_data.csv", index=False)
    print(f"Saved CSV to {output_dir / 'lr_sweep_data.csv'}")


def plot_results(
    all_data: dict[str, dict[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    """Create a 1x2 figure with the two metric plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    fig.suptitle(
        "Learning rate sweep: GRPO experiments",
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
                linewidth=2,
                markersize=meta["markersize"],
                alpha=0.9,
            )

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
        ncol=len(RUN_NAMES),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "results" / "lr_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching runs from {PROJECT}...")
    all_data = fetch_wandb_data()

    if not all_data:
        print("No runs found!")
        return

    save_csv(all_data, output_dir)
    plot_results(all_data, output_dir / "lr_sweep.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot std dev normalization ablation results for GRPO experiments, fetched from W&B.

Creates a single-row figure with 3 subplots:
1. Eval Reward
2. Eval Format Reward
3. Gradient Norm

Compares: with vs. without std dev normalization of advantages.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT = "garg-aayush/grpo"
RUN_NAMES = ["len_norm_mean", "std_dev"]

DISPLAY_NAMES = {
    "len_norm_mean": "With std normalization",
    "std_dev": "No std normalization",
}

COLORS = {
    "len_norm_mean": "#E69F00",   # Orange
    "std_dev": "#56B4E9",         # Sky blue
}

LINE_WIDTHS = {
    "len_norm_mean": 2,
    "std_dev": 1.5,
}

METRICS = {
    "eval/reward": {
        "ylabel": "Reward Accuracy",
        "title": "Eval Reward Accuracy",
        "style": "sparse",
    },
    "eval/format_reward": {
        "ylabel": "Format Reward",
        "title": "Eval Format Reward",
        "style": "sparse",
    },
    "train/grad_norm": {
        "ylabel": "Gradient Norm",
        "title": "Gradient Norm",
        "style": "dense",
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
    pd.DataFrame(rows).to_csv(output_dir / "std_dev_normalization_data.csv", index=False)
    print(f"Saved CSV to {output_dir / 'std_dev_normalization_data.csv'}")


def plot_results(
    all_data: dict[str, dict[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    """Create a 1x3 figure with the three metric plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fig.suptitle(
        "Std dev normalization ablation: GRPO experiments",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, (metric_key, meta) in zip(axes, METRICS.items()):
        for run_name in RUN_NAMES:
            if run_name not in all_data or metric_key not in all_data[run_name]:
                continue
            df = all_data[run_name][metric_key]
            if meta["style"] == "sparse":
                ax.plot(
                    df["step"],
                    df["value"],
                    marker="o",
                    linestyle="-",
                    label=DISPLAY_NAMES[run_name],
                    color=COLORS[run_name],
                    linewidth=LINE_WIDTHS[run_name],
                    markersize=3,
                    alpha=0.9,
                )
            else:
                ax.plot(
                    df["step"],
                    df["value"],
                    linestyle="-",
                    label=DISPLAY_NAMES[run_name],
                    color=COLORS[run_name],
                    linewidth=LINE_WIDTHS[run_name],
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
    output_dir = Path(__file__).resolve().parent.parent.parent / "results" / "std_dev"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching runs from {PROJECT}...")
    all_data = fetch_wandb_data()

    if not all_data:
        print("No runs found!")
        return

    save_csv(all_data, output_dir)
    plot_results(all_data, output_dir / "std_dev_normalization.png")


if __name__ == "__main__":
    main()

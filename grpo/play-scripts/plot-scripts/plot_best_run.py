#!/usr/bin/env python3
"""
Plot eval reward accuracy and mean response length for the best GRPO run:
  off_policy_full_e1_tb256_ga64 (on-policy, lr=3e-5, grpo_clip)

Also annotates the best and last checkpoint eval accuracy on the reward plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT = "garg-aayush/grpo"
RUN_NAME = "off_policy_full_e1_tb256_ga64"
COLOR = "#E69F00"  # Orange (colorblind-friendly)

METRICS = ["eval/reward", "train/mean_response_length"]


def fetch_wandb_data() -> tuple[dict[str, pd.DataFrame], float, int, float, int]:
    """
    Fetch metric histories from W&B.

    Returns:
        data: dict metric_key -> DataFrame(step, value)
        best_reward: best eval/reward value
        best_step: grpo_step at which best_reward occurred
        last_reward: eval/reward value at the last eval step
        last_step: last grpo_step that has an eval/reward
    """
    api = wandb.Api()
    runs = api.runs(PROJECT)

    target_run = None
    for run in runs:
        if run.name == RUN_NAME:
            target_run = run
            break

    if target_run is None:
        raise ValueError(f"Run '{RUN_NAME}' not found in project {PROJECT}")

    print(f"Fetching run: {target_run.name} (id: {target_run.id})")

    data: dict[str, pd.DataFrame] = {}
    for metric_key in METRICS:
        history = target_run.history(keys=[metric_key, "grpo_step"])
        if history.empty:
            print(f"  Warning: no data for {metric_key}")
            continue
        df = history.dropna(subset=[metric_key, "grpo_step"])
        df = df.rename(columns={"grpo_step": "step", metric_key: "value"})
        df = df[["step", "value"]].sort_values("step").reset_index(drop=True)
        data[metric_key] = df
        print(f"  {metric_key}: {len(df)} points")

    # Derive best and last checkpoint stats from eval/reward series
    eval_df = data["eval/reward"]
    best_idx = eval_df["value"].idxmax()
    best_reward = float(eval_df.loc[best_idx, "value"])
    best_step = int(eval_df.loc[best_idx, "step"])

    last_idx = eval_df["step"].idxmax()
    last_reward = float(eval_df.loc[last_idx, "value"])
    last_step = int(eval_df.loc[last_idx, "step"])

    return data, best_reward, best_step, last_reward, last_step


def save_csv(
    data: dict[str, pd.DataFrame],
    best_reward: float,
    best_step: int,
    last_reward: float,
    last_step: int,
    output_dir: Path,
) -> None:
    """Save fetched metric data and checkpoint summary to CSV."""
    rows = []
    for metric_key, df in data.items():
        for _, row in df.iterrows():
            rows.append({
                "run_name": RUN_NAME,
                "metric": metric_key,
                "step": int(row["step"]),
                "value": row["value"],
            })
    csv_path = output_dir / "best_run_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved data CSV to {csv_path}")

    ckpt_path = output_dir / "best_run_checkpoints.csv"
    pd.DataFrame([
        {"checkpoint": "best", "grpo_step": best_step, "eval_reward": best_reward},
        {"checkpoint": "last", "grpo_step": last_step, "eval_reward": last_reward},
    ]).to_csv(ckpt_path, index=False)
    print(f"Saved checkpoint CSV to {ckpt_path}")


def plot_results(
    data: dict[str, pd.DataFrame],
    best_reward: float,
    best_step: int,
    last_reward: float,
    last_step: int,
    output_path: Path,
) -> None:
    """Create a 1x2 figure: eval reward (with checkpoint markers) and mean response length."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    fig.suptitle(
        "Best GRPO run",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # --- Subplot 1: Eval Reward Accuracy ---
    ax = axes[0]
    if "eval/reward" in data:
        df = data["eval/reward"]
        ax.plot(
            df["step"],
            df["value"],
            "o-",
            color=COLOR,
            linewidth=2,
            markersize=4,
            alpha=0.9,
            label="eval/reward",
            zorder=3,
        )

        # Best checkpoint: vertical dashed line + horizontal annotation
        ax.axvline(best_step, color="#CC79A7", linestyle="--", linewidth=1.2, zorder=2)
        ax.axhline(best_reward, color="#CC79A7", linestyle=":", linewidth=1.0, zorder=2)
        ax.scatter(
            [best_step], [best_reward],
            marker="*", s=120, color="#CC79A7", zorder=4,
            label=f"best ckpt  step={best_step}  acc={best_reward:.4f}",
        )

        # Last checkpoint: vertical dashed line + horizontal annotation
        if last_step != best_step:
            ax.axvline(last_step, color="#56B4E9", linestyle="--", linewidth=1.2, zorder=2)
            ax.axhline(last_reward, color="#56B4E9", linestyle=":", linewidth=1.0, zorder=2)
        ax.scatter(
            [last_step], [last_reward],
            marker="D", s=60, color="#56B4E9", zorder=4,
            label=f"last ckpt   step={last_step}  acc={last_reward:.4f}",
        )

    ax.set_xlabel("GRPO Step", fontsize=10)
    ax.set_ylabel("Reward Accuracy", fontsize=10)
    ax.set_title("Eval Reward Accuracy", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=8, loc="lower right")

    # --- Subplot 2: Mean Response Length ---
    ax = axes[1]
    if "train/mean_response_length" in data:
        df = data["train/mean_response_length"]
        ax.plot(
            df["step"],
            df["value"],
            "-",
            color=COLOR,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("GRPO Step", fontsize=10)
    ax.set_ylabel("Mean Response Length (tokens)", fontsize=10)
    ax.set_title("Mean Response Length", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "results" / "best_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching run '{RUN_NAME}' from {PROJECT}...")
    data, best_reward, best_step, last_reward, last_step = fetch_wandb_data()

    print(f"\nCheckpoint summary:")
    print(f"  Best: step={best_step}, eval/reward={best_reward:.4f}")
    print(f"  Last: step={last_step}, eval/reward={last_reward:.4f}")

    save_csv(data, best_reward, best_step, last_reward, last_step, output_dir)
    plot_results(data, best_reward, best_step, last_reward, last_step, output_dir / "best_run.png")


if __name__ == "__main__":
    main()

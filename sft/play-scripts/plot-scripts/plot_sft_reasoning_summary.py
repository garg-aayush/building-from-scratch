#!/usr/bin/env python3
"""
Plot training curve for the best Reasoning SFT run (run_filtered_2epoch).

Shows:
- Baseline as horizontal dashed line
- Accuracy at each evaluation point during training
- Final accuracy on full validation set (star marker)

For use in the "What I Built" section of the blog.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Final results from Notes.md
FINAL_RESULTS = {
    "baseline": {"reward_acc": 0.0288, "format_acc": 0.1438},
    "run_filtered_2epoch": {"reward_acc": 0.5336, "format_acc": 0.9926},
}


def fetch_run_data(project: str = "garg-aayush/sft", run_name: str = "run_filtered_2epoch") -> pd.DataFrame:
    """Fetch training curve data from W&B."""
    api = wandb.Api()
    runs = api.runs(project)

    for run in runs:
        if run.name == run_name:
            print(f"Fetching run: {run.name}")
            history = run.history(keys=["eval/avg_acc", "eval/avg_format_acc", "_step"])
            
            if history.empty:
                print(f"Warning: No eval data found for {run.name}")
                return pd.DataFrame()

            df = history.dropna(subset=["eval/avg_acc", "eval/avg_format_acc"])
            df = df.rename(columns={
                "_step": "step",
                "eval/avg_acc": "reward_acc",
                "eval/avg_format_acc": "format_acc",
            })
            df = df.sort_values("step").reset_index(drop=True)
            print(f"Found {len(df)} eval points")
            return df

    print(f"Run {run_name} not found!")
    return pd.DataFrame()


def plot_training_curve(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """
    Create a line plot showing training progression.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    baseline_reward = FINAL_RESULTS["baseline"]["reward_acc"]
    baseline_format = FINAL_RESULTS["baseline"]["format_acc"]
    final_reward = FINAL_RESULTS["run_filtered_2epoch"]["reward_acc"]
    final_format = FINAL_RESULTS["run_filtered_2epoch"]["format_acc"]

    # === Left: Reward Accuracy ===
    # Baseline line
    ax1.axhline(
        y=baseline_reward,
        color="#888888",
        linestyle="--",
        linewidth=2,
        label=f"Baseline ({baseline_reward:.1%})",
    )

    # Training curve
    ax1.plot(
        df["step"],
        df["reward_acc"],
        "o-",
        color="#4CAF50",
        linewidth=2.5,
        markersize=7,
        label="eval on 1K subset",
    )

    # Final accuracy (star)
    final_step = df["step"].max()
    ax1.scatter(
        [final_step],
        [final_reward],
        marker="*",
        s=300,
        color="#E91E63",
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
        label=f"eval on full val. set ({final_reward:.1%})",
    )

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Reward Accuracy", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.65)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # === Right: Format Accuracy ===
    # Baseline line
    ax2.axhline(
        y=baseline_format,
        color="#888888",
        linestyle="--",
        linewidth=2,
        label=f"Baseline ({baseline_format:.1%})",
    )

    # Training curve
    ax2.plot(
        df["step"],
        df["format_acc"],
        "o-",
        color="#2196F3",
        linewidth=2.5,
        markersize=7,
        label="eval on 1K subset",
    )

    # Final accuracy (star)
    ax2.scatter(
        [final_step],
        [final_format],
        marker="*",
        s=300,
        color="#E91E63",
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
        label=f"eval on full val. set ({final_reward:.1%})",
    )

    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Format Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Main title
    fig.suptitle(
        "Reasoning SFT: Qwen2.5-Math-1.5B → Fine-tuned on Reasoning Traces",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def main():
    output_dir = Path(__file__).parent.parent.parent / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sft_reasoning_summary.png"

    # Fetch data from W&B
    df = fetch_run_data()
    
    if df.empty:
        print("No data to plot!")
        return

    plot_training_curve(df, output_path)


if __name__ == "__main__":
    main()

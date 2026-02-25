#!/usr/bin/env python3
"""
Plot Expert Iteration evaluation accuracy fetched from W&B.

This script:
1. Fetches runs from the W&B project
2. Downloads eval_ei/avg_acc metrics
3. Saves the data to a CSV file
4. Creates a comparison plot of accuracy across iterations
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Comparison presets: each contains runs, display names, colors, and title
COMPARISONS = {
    "lr_sweep": {
        "runs": ["lrsweep_7e-5_D512_G5_R4", "lrsweep_1e-5_D512_G5_R4", "lrsweep_adaptive_D512_G5_R4"],
        "display_names": {
            "lrsweep_7e-5_D512_G5_R4": "LR 7e-5",
            "lrsweep_1e-5_D512_G5_R4": "LR 1e-5",
            "lrsweep_adaptive_D512_G5_R4": "Adaptive LR",
        },
        "colors": {
            "lrsweep_7e-5_D512_G5_R4": "#00BCD4",      # Cyan
            "lrsweep_1e-5_D512_G5_R4": "#FF9800",      # Orange
            "lrsweep_adaptive_D512_G5_R4": "#E91E63",  # Pink/Magenta
        },
        "title": "Expert Iteration: Accuracy Comparison (LR Sweep)",
        "output_prefix": "lr_sweep",
    },
    "sampling_strategy": {
        "runs": ["sampling_one_D512_G5_R4", "sampling_multi_D512_G5_R4"],
        "display_names": {
            "sampling_one_D512_G5_R4": "Single Trace",
            "sampling_multi_D512_G5_R4": "Multi Trace",
        },
        "colors": {
            "sampling_one_D512_G5_R4": "#F4A460",      # Sandy brown
            "sampling_multi_D512_G5_R4": "#2E8B57",    # Sea green
        },
        "title": "Expert Iteration: Single vs Multi-Trace Sampling",
        "output_prefix": "sampling_strategy",
    },
}


def fetch_wandb_runs(
    project: str,
    run_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch runs from W&B and return history data.

    Args:
        project: W&B project path (entity/project)
        run_names: List of run names to fetch. If None, fetches all runs.

    Returns:
        Dictionary mapping run names to DataFrames with metrics history.
    """
    api = wandb.Api()
    runs = api.runs(project)

    run_data = {}
    for run in runs:
        # Skip if run_names specified and this run not in list
        if run_names and run.name not in run_names:
            continue

        print(f"Fetching run: {run.name} (id: {run.id}, state: {run.state})")

        # Get history for eval_ei/avg_acc metric
        history = run.history(keys=["eval_ei/avg_acc", "_step"])

        if history.empty:
            print(f"  Warning: No eval data found for {run.name}")
            continue

        # Clean up the data
        df = history.dropna(subset=["eval_ei/avg_acc"])
        df = df.rename(columns={
            "_step": "step",
            "eval_ei/avg_acc": "avg_acc",
        })
        df["run_name"] = run.name
        df = df.sort_values("step").reset_index(drop=True)
        # Create ei_step locally (1-indexed based on evaluation order)
        df["ei_step"] = range(0, len(df))

        run_data[run.name] = df
        print(f"  Found {len(df)} eval points")

    return run_data


def save_to_csv(run_data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save all run data to a single CSV file."""
    all_data = pd.concat(run_data.values(), ignore_index=True)
    all_data = all_data[["run_name", "step", "ei_step", "avg_acc"]]
    all_data.to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")


def plot_results(
    run_data: dict[str, pd.DataFrame],
    comparison_config: dict,
    output_path: Path | None = None,
) -> None:
    """
    Create a plot comparing accuracy across different runs.

    Args:
        run_data: Dictionary mapping run names to DataFrames.
        comparison_config: Config dict with display_names, colors, and title.
        output_path: Path to save the figure. If None, only displays.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main title
    fig.suptitle(
        comparison_config["title"],
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    colors = comparison_config["colors"]
    display_names = comparison_config["display_names"]

    for run_name, df in sorted(run_data.items()):
        color = colors.get(run_name, "#333333")
        display_name = display_names.get(run_name, run_name)

        # Plot training curve against ei_step
        ax.plot(
            df["ei_step"],
            df["avg_acc"],
            "o-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=5,
            alpha=0.9,
        )

        # Annotate final value
        if len(df) > 0:
            final_ei_step = df["ei_step"].iloc[-1]
            final_val = df["avg_acc"].iloc[-1]
            ax.annotate(
                f"{final_val:.3f}",
                xy=(final_ei_step, final_val),
                xytext=(0, -12),
                textcoords="offset points",
                fontsize=9,
                color=color,
                ha="center",
                fontweight="bold",
            )

    ax.set_xlabel("Expert Iteration", fontsize=10)
    ax.set_ylabel("Average Accuracy", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Expert Iteration results from W&B")
    parser.add_argument(
        "--comparison",
        type=str,
        choices=list(COMPARISONS.keys()),
        default="lr_sweep",
        help=f"Comparison preset to use: {list(COMPARISONS.keys())}",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="garg-aayush/expert-iter",
        help="W&B project path (default: garg-aayush/expert-iter)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for CSV and plot",
    )
    args = parser.parse_args()

    # Get comparison config
    config = COMPARISONS[args.comparison]

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data from W&B
    print(f"Fetching runs from {args.project}...")
    print(f"Using comparison preset: {args.comparison}")
    run_data = fetch_wandb_runs(args.project, config["runs"])

    if not run_data:
        print("No runs found!")
        return

    # Save to CSV
    csv_path = args.output_dir / f"{config['output_prefix']}_ei_accuracy.csv"
    save_to_csv(run_data, csv_path)

    # Plot results
    plot_path = args.output_dir / f"{config['output_prefix']}_ei_acc.png"
    plot_results(run_data, config, plot_path)


if __name__ == "__main__":
    main()

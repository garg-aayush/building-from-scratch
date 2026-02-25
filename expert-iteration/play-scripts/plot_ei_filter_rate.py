#!/usr/bin/env python3
"""
Plot Expert Iteration filter rate analysis.

Shows how the percentage of correct rollouts (filter rate) evolves across
EI iterations for different configurations.

Filter Rate = num_filtered_examples / (D × R)

Expected pattern: Filter rate should increase as the model improves (virtuous cycle).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Run configurations: run_name -> (D, R)
RUN_CONFIGS = {
    "run_D512_G5_R2": (512, 2),
    "run_D1024_G5_R2": (1024, 2),
    "run_D2048_G5_R2": (2048, 2),
    "run_D512_G5_R4": (512, 4),
    "run_D1024_G5_R4": (1024, 4),
    "run_D2048_G5_R4": (2048, 4),
}

# Run name mappings for display
RUN_DISPLAY_NAMES = {
    "run_D512_G5_R2": "D=512, R=2",
    "run_D1024_G5_R2": "D=1024, R=2",
    "run_D2048_G5_R2": "D=2048, R=2",
    "run_D512_G5_R4": "D=512, R=4",
    "run_D1024_G5_R4": "D=1024, R=4",
    "run_D2048_G5_R4": "D=2048, R=4",
}

# Colors for each run
RUN_COLORS = {
    "run_D512_G5_R2": "#9E9E9E",      # Gray
    "run_D1024_G5_R2": "#2196F3",     # Blue
    "run_D2048_G5_R2": "#E91E63",     # Pink
    "run_D512_G5_R4": "#8BC34A",      # Light green
    "run_D1024_G5_R4": "#FF9800",     # Orange
    "run_D2048_G5_R4": "#9C27B0",     # Purple
}

ALL_RUNS = list(RUN_CONFIGS.keys())


def fetch_wandb_runs(
    project: str,
    run_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch runs from W&B and return history data with filter rate."""
    api = wandb.Api()
    runs = api.runs(project)

    run_data = {}
    for run in runs:
        if run_names and run.name not in run_names:
            continue

        if run.name not in RUN_CONFIGS:
            continue

        print(f"Fetching run: {run.name} (id: {run.id}, state: {run.state})")

        history = run.history(keys=["eval_ei/num_filtered_examples", "_step"])

        if history.empty:
            print(f"  Warning: No data found for {run.name}")
            continue

        df = history.dropna(subset=["eval_ei/num_filtered_examples"])
        df = df.rename(columns={
            "_step": "step",
            "eval_ei/num_filtered_examples": "num_filtered",
        })
        df["run_name"] = run.name
        df = df.sort_values("step").reset_index(drop=True)
        df["ei_step"] = range(0, len(df))

        # Calculate filter rate = num_filtered / (D × R)
        D, R = RUN_CONFIGS[run.name]
        total_rollouts = D * R
        df["filter_rate"] = df["num_filtered"] / total_rollouts

        run_data[run.name] = df
        print(f"  Found {len(df)} points, D={D}, R={R}, total_rollouts={total_rollouts}")

    return run_data


def save_to_csv(run_data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save all run data to a single CSV file."""
    all_data = pd.concat(run_data.values(), ignore_index=True)
    all_data = all_data[["run_name", "step", "ei_step", "num_filtered", "filter_rate"]]
    all_data.to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")


def plot_results(
    run_data: dict[str, pd.DataFrame],
    output_path: Path | None = None,
) -> None:
    """Create two subplots: total examples and filter rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Main title
    fig.suptitle(
        "Expert Iteration: Filter Rate Analysis",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Sort runs by D then R for consistent ordering
    sorted_runs = sorted(run_data.keys(), key=lambda x: (
        int(x.split("_D")[1].split("_")[0]),
        int(x.split("_R")[1]),
    ))

    for run_name in sorted_runs:
        df = run_data[run_name]
        color = RUN_COLORS.get(run_name, "#333333")
        display_name = RUN_DISPLAY_NAMES.get(run_name, run_name)

        # Left plot: Total filtered examples
        ax1.plot(
            df["ei_step"],
            df["num_filtered"],
            "o-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

        # Right plot: Filter rate
        ax2.plot(
            df["ei_step"],
            df["filter_rate"],
            "o-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    # Left plot styling
    ax1.set_xlabel("Expert Iteration", fontsize=11)
    ax1.set_ylabel("Number of Filtered Examples", fontsize=11)
    ax1.set_title("Total Correct Rollouts", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right plot styling
    ax2.set_xlabel("Expert Iteration", fontsize=11)
    ax2.set_ylabel("Filter Rate (Correct / Total)", fontsize=11)
    ax2.set_title("Filter Rate", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Set x-axis limits and integer ticks
    ax1.set_xlim(-0.2, 5.2)
    ax2.set_xlim(-0.2, 5.2)
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticks([0, 1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Expert Iteration filter rate")
    parser.add_argument(
        "--project",
        type=str,
        default="garg-aayush/expert-iter",
        help="W&B project path",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=ALL_RUNS,
        help="Run names to fetch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for CSV and plot",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="ei_filter_rate_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="ei_filter_rate_plot.png",
        help="Plot output filename",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching runs from {args.project}...")
    run_data = fetch_wandb_runs(args.project, args.runs)

    if not run_data:
        print("No runs found!")
        return

    csv_path = args.output_dir / args.csv_name
    save_to_csv(run_data, csv_path)

    plot_path = args.output_dir / args.plot_name
    plot_results(run_data, plot_path)


if __name__ == "__main__":
    main()

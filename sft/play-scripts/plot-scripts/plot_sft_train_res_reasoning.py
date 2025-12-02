#!/usr/bin/env python3
"""
Plot SFT training results fetched from W&B.

This script:
1. Fetches runs from the W&B project 'garg-aayush/sft'
2. Downloads eval/avg_acc and eval/avg_format_acc metrics
3. Saves the data to a CSV file
4. Creates a single figure with 2 subplots (Reward Accuracy and Format Accuracy)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Run name mappings for display
RUN_DISPLAY_NAMES = {
    "run_all": "run_all",
    "run_filtered": "run_filtered", 
    "run_filtered_res_len": "run_filtered_res_len",
    "run_filtered_2epoch": "run_filtered_2epoch",
}

# Colors for each run (colorblind-friendly palette)
RUN_COLORS = {
    "run_all": "#E69F00",           # Orange
    "run_filtered": "#56B4E9",       # Sky blue
    "run_filtered_res_len": "#009E73", # Bluish green
    "run_filtered_2epoch": "#CC79A7",  # Reddish purple
}

# Final results from Notes.md (for reference annotations)
FINAL_RESULTS = {
    "baseline": {"reward_acc": 0.0288, "format_acc": 0.1438},
    "run_all": {"reward_acc": 0.4214, "format_acc": 0.9924},
    "run_filtered": {"reward_acc": 0.5204, "format_acc": 0.9906},
    "run_filtered_res_len": {"reward_acc": 0.5106, "format_acc": 0.9898},
    "run_filtered_2epoch": {"reward_acc": 0.5336, "format_acc": 0.9926},
}

# Run descriptions for the table
RUN_DESCRIPTIONS = {
    "baseline": "Untrained Qwen2.5-Math-1.5B",
    "run_all": "Full ~4.8K examples",
    "run_filtered": "Filtered ~3.6K (correct answers only)",
    "run_filtered_res_len": "Filtered, no per-token loss",
    "run_filtered_2epoch": "Filtered, 2 epochs",
}


def fetch_wandb_runs(
    project: str = "garg-aayush/sft",
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

        # Get history for eval metrics
        history = run.history(keys=["eval/avg_acc", "eval/avg_format_acc", "_step"])

        if history.empty:
            print(f"  Warning: No eval data found for {run.name}")
            continue

        # Clean up the data
        df = history.dropna(subset=["eval/avg_acc", "eval/avg_format_acc"])
        df = df.rename(columns={
            "_step": "step",
            "eval/avg_acc": "reward_acc",
            "eval/avg_format_acc": "format_acc",
        })
        df["run_name"] = run.name
        df = df.sort_values("step").reset_index(drop=True)

        run_data[run.name] = df
        print(f"  Found {len(df)} eval points")

    return run_data


def save_to_csv(run_data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save all run data to a single CSV file."""
    all_data = pd.concat(run_data.values(), ignore_index=True)
    all_data = all_data[["run_name", "step", "reward_acc", "format_acc"]]
    all_data.to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")


def format_step(x, pos):
    """Format tick as 10, 20, etc. for step_eval axis."""
    return f"{int(x)}"


def plot_results(
    run_data: dict[str, pd.DataFrame],
    output_path: Path | None = None,
    show_baseline: bool = True,
) -> None:
    """
    Create a figure with reward accuracy plot and a results table.

    Args:
        run_data: Dictionary mapping run names to DataFrames.
        output_path: Path to save the figure. If None, only displays.
        show_baseline: Whether to show baseline reference line.
    """
    fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    # Main title and subtitle
    fig.suptitle(
        "SFT on Qwen2.5-Math-1.5B with Reasoning Traces",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Lines: intermediate eval on 1K validation subset  |  ★ Final: eval on full ~5K validation set",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555555",
    )

    # === Left: Reward Accuracy Plot ===
    if show_baseline:
        baseline_val = FINAL_RESULTS["baseline"]["reward_acc"]
        ax_plot.axhline(
            y=baseline_val,
            color="#888888",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Baseline ({baseline_val:.3f})",
        )

    for run_name, df in sorted(run_data.items()):
        color = RUN_COLORS.get(run_name, "#333333")
        display_name = RUN_DISPLAY_NAMES.get(run_name, run_name)

        # Plot training curve (1K subset evals)
        ax_plot.plot(
            df["step"],
            df["reward_acc"],
            "o-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=5,
            alpha=0.9,
        )

        # Plot final accuracy on full validation set (star marker)
        if run_name in FINAL_RESULTS:
            final_step = df["step"].max()
            final_val = FINAL_RESULTS[run_name]["reward_acc"]
            ax_plot.scatter(
                [final_step],
                [final_val],
                marker="*",
                s=200,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=10,
            )

    ax_plot.set_xlabel("step", fontsize=10)
    ax_plot.set_ylabel("Accuracy", fontsize=10)
    ax_plot.set_title("Reward Accuracy", fontsize=11, fontweight="bold")
    ax_plot.legend(loc="lower right", fontsize=8)
    ax_plot.grid(True, alpha=0.3)
    ax_plot.tick_params(axis="both", labelsize=9)
    ax_plot.set_ylim(0, 0.6)

    # === Right: Results Table ===
    ax_table.axis("off")

    # Build table data
    table_data = []
    cell_colors = []
    for run_name in ["baseline", "run_all", "run_filtered", "run_filtered_res_len", "run_filtered_2epoch"]:
        if run_name in FINAL_RESULTS:
            display_name = RUN_DISPLAY_NAMES.get(run_name, run_name)
            reward = FINAL_RESULTS[run_name]["reward_acc"]
            fmt = FINAL_RESULTS[run_name]["format_acc"]
            table_data.append([display_name, f"{reward:.4f}", f"{fmt:.4f}"])
            
            color = RUN_COLORS.get(run_name, "#888888")
            # Light version of the color for cell background
            cell_colors.append([color + "30", "#FFFFFF", "#FFFFFF"])

    col_labels = ["Run", "Reward Accuracy", "Format Accuracy"]
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
        colColours=["#E0E0E0"] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Style header row
    for j in range(3):
        table[(0, j)].set_text_props(fontweight="bold")

    # Make reward accuracy bold for run_filtered (row 3) and run_filtered_2epoch (row 5)
    # Row indices: 0=header, 1=baseline, 2=run_all, 3=run_filtered, 4=run_filtered_res_len, 5=run_filtered_2epoch
    table[(3, 1)].set_text_props(fontweight="bold")
    table[(5, 1)].set_text_props(fontweight="bold")

    # Add note below the table
    ax_table.text(
        0.5, 0.25,
        "*Final accuracy evaluated on the full validation set (~5K examples)",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="#555555",
        transform=ax_table.transAxes,
    )

    # Run descriptions below the note (run names in bold using separate text elements)
    descriptions = [
        ("baseline", "Untrained Qwen2.5-Math-1.5B"),
        ("run_all", "Full ~4.8K examples"),
        ("run_filtered", "Filtered ~3.6K (correct answers only)"),
        ("run_filtered_res_len", "Filtered, no per-token loss"),
        ("run_filtered_2epoch", "Filtered, 2 epochs"),
    ]
    y_start = 0.18
    line_height = 0.028
    desc_x_offset = 0.35  # Offset for description to avoid overlap with long run names
    for i, (run_name, desc) in enumerate(descriptions):
        y_pos = y_start - i * line_height
        # Bold run name
        ax_table.text(
            0.08, y_pos,
            f"{run_name}:",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#444444",
            transform=ax_table.transAxes,
        )
        # Regular description (positioned after the run name)
        ax_table.text(
            0.08 + desc_x_offset, y_pos,
            desc,
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
            transform=ax_table.transAxes,
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    # plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot SFT training results from W&B")
    parser.add_argument(
        "--project",
        type=str,
        default="garg-aayush/sft",
        help="W&B project path (default: garg-aayush/sft)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=["run_all", "run_filtered", "run_filtered_res_len", "run_filtered_2epoch"],
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
        default="sft_training_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="sft_training_results.png",
        help="Plot output filename",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Don't show baseline reference line",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data from W&B
    print(f"Fetching runs from {args.project}...")
    run_data = fetch_wandb_runs(args.project, args.runs)

    if not run_data:
        print("No runs found!")
        return

    # 

    # Plot results
    plot_path = args.output_dir / args.plot_name
    plot_results(run_data, plot_path, show_baseline=not args.no_baseline)


if __name__ == "__main__":
    main()

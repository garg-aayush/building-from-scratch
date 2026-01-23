#!/usr/bin/env python3
"""
Plot Expert Iteration accuracy comparing single vs multi-epoch training.

Compares 1 epoch vs 2 epoch training for best configurations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Run name mappings for display
RUN_DISPLAY_NAMES = {
    "run_D512_G5_R4": "D=512, 1 epoch",
    "run_D1024_G5_R4": "D=1024, 1 epoch",
    "run_Ep2_D512_G5_R4": "D=512, 2 epochs",
    "run_Ep2_D1024_G5_R4": "D=1024, 2 epochs",
}

# Colors for each run (grouped by epochs)
RUN_COLORS = {
    # 1 epoch (lighter)
    "run_D512_G5_R4": "#8BC34A",      # Light green
    "run_D1024_G5_R4": "#FF9800",     # Orange
    # 2 epochs (darker/bolder)
    "run_Ep2_D512_G5_R4": "#388E3C",  # Dark green
    "run_Ep2_D1024_G5_R4": "#E65100", # Dark orange
}

ALL_RUNS = [
    "run_D512_G5_R4", "run_D1024_G5_R4",
    "run_Ep2_D512_G5_R4", "run_Ep2_D1024_G5_R4",
]


def fetch_wandb_runs(
    project: str,
    run_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch runs from W&B and return history data."""
    api = wandb.Api()
    runs = api.runs(project)

    run_data = {}
    for run in runs:
        if run_names and run.name not in run_names:
            continue

        print(f"Fetching run: {run.name} (id: {run.id}, state: {run.state})")

        history = run.history(keys=["eval_ei/avg_acc", "_step"])

        if history.empty:
            print(f"  Warning: No eval data found for {run.name}")
            continue

        df = history.dropna(subset=["eval_ei/avg_acc"])
        df = df.rename(columns={
            "_step": "step",
            "eval_ei/avg_acc": "avg_acc",
        })
        df["run_name"] = run.name
        df = df.sort_values("step").reset_index(drop=True)
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
    output_path: Path | None = None,
) -> None:
    """Create plot with accuracy curves and adjacent table."""
    fig, (ax_plot, ax_table) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # Main title
    fig.suptitle(
        "Expert Iteration: Multi-Epoch Training Comparison",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Sort: 1 epoch first, then 2 epochs; within each, by D
    def sort_key(name):
        is_ep2 = "Ep2" in name
        d_val = int(name.split("_D")[1].split("_")[0])
        return (is_ep2, d_val)

    sorted_runs = sorted(run_data.keys(), key=sort_key)

    for run_name in sorted_runs:
        df = run_data[run_name]
        color = RUN_COLORS.get(run_name, "#333333")
        display_name = RUN_DISPLAY_NAMES.get(run_name, run_name)

        # Use dashed line for 1 epoch, solid for 2 epochs
        linestyle = "-" if "Ep2" in run_name else "--"

        ax_plot.plot(
            df["ei_step"],
            df["avg_acc"],
            marker="o",
            linestyle=linestyle,
            label=display_name,
            color=color,
            linewidth=2,
            markersize=5,
            alpha=0.9,
        )

    ax_plot.set_xlabel("Expert Iteration", fontsize=10)
    ax_plot.set_ylabel("Average Accuracy", fontsize=10)
    ax_plot.legend(loc="lower right", fontsize=9)
    ax_plot.grid(True, alpha=0.3)
    ax_plot.tick_params(axis="both", labelsize=9)
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.set_xlim(-0.2, 5.2)
    ax_plot.set_xticks([0, 1, 2, 3, 4, 5])

    # === Right: Results Table ===
    ax_table.axis("off")

    # Build table data
    table_data = []
    cell_colors = []

    for run_name in sorted_runs:
        df = run_data[run_name]
        display_name = RUN_DISPLAY_NAMES.get(run_name, run_name)
        final_acc = df["avg_acc"].iloc[-1] if len(df) > 0 else 0.0
        table_data.append([display_name, f"{final_acc:.4f}"])

        color = RUN_COLORS.get(run_name, "#888888")
        cell_colors.append([color + "30", "#FFFFFF"])

    col_labels = ["Configuration", "Final Accuracy"]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
        colColours=["#E0E0E0"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.8)

    # Style header row
    for j in range(2):
        table[(0, j)].set_text_props(fontweight="bold")

    # Add note below table
    ax_table.text(
        0.5, 0.18,
        "Dashed lines: 1 epoch | Solid lines: 2 epochs per iteration",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="#555555",
        transform=ax_table.transAxes,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Expert Iteration multi-epoch comparison")
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
        default="ei_multiepoch_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="ei_multiepoch_plot.png",
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

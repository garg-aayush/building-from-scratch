#!/usr/bin/env python3
"""
Summarize all runs in the garg-aayush/grpo W&B project.

For each run, prints:
  - run name
  - last grpo_step logged
  - best eval/reward value
  - grpo_step at which best eval/reward occurred
"""

import pandas as pd
import wandb

PROJECT = "garg-aayush/grpo"
EVAL_METRIC = "eval/reward"


def summarize_runs() -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(PROJECT)

    rows = []
    for run in runs:
        print(f"Processing: {run.name} (id: {run.id}, state: {run.state})")

        # Fetch all grpo_step values to find the last one
        step_history = run.history(keys=["grpo_step"])
        if step_history.empty or step_history["grpo_step"].isna().all():
            print(f"  Warning: no grpo_step data, skipping")
            continue

        last_grpo_step = int(step_history["grpo_step"].dropna().max())

        # Fetch eval/reward with grpo_step
        eval_history = run.history(keys=[EVAL_METRIC, "grpo_step"])
        if eval_history.empty or EVAL_METRIC not in eval_history.columns:
            best_eval = None
            best_grpo_step = None
        else:
            eval_df = eval_history.dropna(subset=[EVAL_METRIC, "grpo_step"])
            if eval_df.empty:
                best_eval = None
                best_grpo_step = None
            else:
                best_idx = eval_df[EVAL_METRIC].idxmax()
                best_eval = eval_df.loc[best_idx, EVAL_METRIC]
                best_grpo_step = int(eval_df.loc[best_idx, "grpo_step"])

        rows.append({
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
            "last_grpo_step": last_grpo_step,
            "best_eval_reward": best_eval,
            "best_eval_grpo_step": best_grpo_step,
        })

    df = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)
    return df


def main():
    print(f"Fetching all runs from {PROJECT}...\n")
    df = summarize_runs()

    if df.empty:
        print("No runs found.")
        return

    # Pretty-print
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 90)
    print("RUN SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))

    # Save CSV
    out_path = "grpo_run_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

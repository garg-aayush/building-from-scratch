#!/usr/bin/env python3
"""Plot accuracy comparison between mask and nomask experiments."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_checkpoint_num(filename: str) -> int:
    """Extract checkpoint number from filename like 'ckpt_1400_gsm8k_accuracy.jsonl'."""
    match = re.search(r'ckpt_(\d+)', filename)
    return int(match.group(1)) if match else 0


def load_gsm8k_accuracy(filepath: Path) -> float:
    with open(filepath) as f:
        return json.load(f)["accuracy"]


def load_mmlu_accuracy(filepath: Path) -> float:
    with open(filepath) as f:
        data = json.load(f)
        return data["all_subjects"]["accuracy"]


def load_sst_accuracy(filepath: Path) -> float:
    with open(filepath) as f:
        return json.load(f)["accuracy"]


def load_alpaca_eval_winrate(filepath: Path) -> float:
    df = pd.read_csv(filepath)
    return df["length_controlled_winrate"].iloc[0]


def load_checkpoint_data(exp_dir: Path) -> dict:
    """Load accuracy data for all checkpoints."""
    data = {"gsm8k": {}, "mmlu": {}, "sst": {}, "alpaca_eval": {}}
    
    # GSM8K
    for f in (exp_dir / "gsm8k").glob("ckpt_*_gsm8k_accuracy.jsonl"):
        ckpt = extract_checkpoint_num(f.name)
        data["gsm8k"][ckpt] = load_gsm8k_accuracy(f)
    
    # MMLU
    for f in (exp_dir / "mmlu").glob("ckpt_*_mmlu_accuracy.jsonl"):
        ckpt = extract_checkpoint_num(f.name)
        data["mmlu"][ckpt] = load_mmlu_accuracy(f)
    
    # SST
    for f in (exp_dir / "sst").glob("ckpt_*_sst_accuracy.jsonl"):
        ckpt = extract_checkpoint_num(f.name)
        data["sst"][ckpt] = load_sst_accuracy(f)
    
    # AlpacaEval
    for f in (exp_dir / "alpaca_eval").glob("ckpt_*_alpaca_eval_leaderboard.csv"):
        ckpt = extract_checkpoint_num(f.name)
        data["alpaca_eval"][ckpt] = load_alpaca_eval_winrate(f)
    
    return data


def load_baseline_data(baseline_dir: Path) -> dict:
    """Load baseline accuracy data."""
    return {
        "gsm8k": load_gsm8k_accuracy(baseline_dir / "baseline_gsm8k_accuracy.jsonl"),
        "mmlu": load_mmlu_accuracy(baseline_dir / "baseline_mmlu_accuracy.jsonl"),
        "sst": load_sst_accuracy(baseline_dir / "baseline_sst_eval_accuracy.jsonl"),
        "alpaca_eval": load_alpaca_eval_winrate(baseline_dir / "baseline_alpaca_eval_leaderboard.csv"),
    }


def format_k(x, pos):
    """Format tick as 1K, 2K, etc."""
    if x == 0:
        return '0'
    return f'{int(x/1000)}K'


def plot_comparison(mask_data: dict, nomask_data: dict, baseline_data: dict, output_path: Path = None):
    """Plot accuracy comparison between mask and nomask experiments."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    
    # Add main title (compact, single line)
    fig.suptitle('SFT: Llama-3.1-8B  |  Training: UltraChat-200K + SafetyLlama  |  Mask vs No-Mask Comparison',
                 fontsize=11, fontweight='bold', y=1.02)
    
    metrics = [
        ("gsm8k", "GSM8K Accuracy", "Accuracy"),
        ("mmlu", "MMLU Accuracy", "Accuracy"),
        ("sst", "SST Accuracy", "Accuracy"),
        ("alpaca_eval", "AlpacaEval Win Rate (vs GPT-4)", "LC Win Rate (%)"),
    ]
    
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        baseline = baseline_data[metric]
        
        # Plot mask experiment
        mask_ckpts = sorted(mask_data[metric].keys())
        mask_values = [mask_data[metric][c] for c in mask_ckpts]
        ax.plot(mask_ckpts, mask_values, 'o-', label='Mask', color='#4CAF50', linewidth=2, markersize=6)
        
        # Plot nomask experiment
        nomask_ckpts = sorted(nomask_data[metric].keys())
        nomask_values = [nomask_data[metric][c] for c in nomask_ckpts]
        ax.plot(nomask_ckpts, nomask_values, 's-', label='No-Mask', color='#2196F3', linewidth=2, markersize=6)
        
        # Plot baseline as horizontal dashed line
        ax.axhline(y=baseline, color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Plot baseline as a point at x=0 with annotation
        ax.scatter([0], [baseline], color='#FF5722', s=60, zorder=5, marker='D', label='Baseline')
        ax.annotate(f'{baseline:.3f}', xy=(0, baseline), xytext=(5, 4), 
                    textcoords='offset points', fontsize=8, color='#FF5722', va='bottom')
        
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=-300)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_k))
        ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def main():
    script_dir = Path(__file__).parent
    mask_dir = script_dir / "mask"
    nomask_dir = script_dir / "nomask"
    baseline_dir = script_dir / "baseline"
    
    print("Loading mask experiment data...")
    mask_data = load_checkpoint_data(mask_dir)
    
    print("Loading nomask experiment data...")
    nomask_data = load_checkpoint_data(nomask_dir)
    
    print("Loading baseline data...")
    baseline_data = load_baseline_data(baseline_dir)
    
    print("\nMask checkpoints:")
    for metric, data in mask_data.items():
        ckpts = sorted(data.keys())
        print(f"  {metric}: {ckpts}")
    
    print("\nNo-Mask checkpoints:")
    for metric, data in nomask_data.items():
        ckpts = sorted(data.keys())
        print(f"  {metric}: {ckpts}")
    
    print("\nBaseline values:")
    for metric, value in baseline_data.items():
        print(f"  {metric}: {value:.4f}")
    
    output_path = script_dir / "eval_comparison_plot.png"
    plot_comparison(mask_data, nomask_data, baseline_data, output_path)


if __name__ == "__main__":
    main()

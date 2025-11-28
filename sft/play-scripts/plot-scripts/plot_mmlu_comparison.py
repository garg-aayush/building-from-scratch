"""
MMLU Evaluation Comparison: Baseline vs Fine-tuned LLaMA 3.1 8B
Generates side-by-side plots for category comparison and subject-level deltas.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set up clean white style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#cccccc',
    'grid.linewidth': 0.5,
    'text.color': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#cccccc',
})

# Colors matching the reference style
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'
RED = '#d62728'

# Load data
def load_data():
    base_path = Path(__file__).parent
    
    with open(base_path / 'baseline/baseline_mmlu_accuracy.jsonl', 'r') as f:
        baseline = json.load(f)
    
    with open(base_path / 'nomask/mmlu/ckpt_6726_mmlu_accuracy.jsonl', 'r') as f:
        finetuned = json.load(f)
    
    return baseline, finetuned

# Category groupings
CATEGORY_GROUPS = {
    "STEM": [
        "abstract algebra", "astronomy", "college biology", "college chemistry", "college computer science",
        "college mathematics", "college physics", "computer security", "conceptual physics", "electrical engineering",
        "elementary mathematics", "high school biology", "high school chemistry", "high school computer science",
        "high school mathematics", "high school physics", "high school statistics", "machine learning"
    ],
    "Humanities": [
        "formal logic", "high school european history", "high school us history", "high school world history",
        "jurisprudence", "logical fallacies", "moral disputes", "moral scenarios", "philosophy", "prehistory", "world religions"
    ],
    "Social\nSciences": [
        "econometrics", "high school geography", "high school government and politics", "high school macroeconomics",
        "high school microeconomics", "high school psychology", "human sexuality", "professional psychology",
        "public relations", "security studies", "sociology", "us foreign policy"
    ],
    "Medicine &\nHealth": [
        "anatomy", "clinical knowledge", "college medicine", "human aging", "medical genetics", "nutrition",
        "professional medicine", "virology"
    ],
    "Business\n& Law": [
        "business ethics", "international law", "management", "marketing", "professional accounting", "professional law"
    ],
    "Other": [
        "global facts", "miscellaneous"
    ]
}

def compute_category_stats(baseline, finetuned):
    """Compute average accuracy per category."""
    category_data = []
    for category, subjects in CATEGORY_GROUPS.items():
        baseline_accs = [baseline[s]['accuracy'] for s in subjects if s in baseline]
        finetuned_accs = [finetuned[s]['accuracy'] for s in subjects if s in finetuned]
        
        if baseline_accs and finetuned_accs:
            category_data.append({
                'category': category,
                'baseline': np.mean(baseline_accs),
                'finetuned': np.mean(finetuned_accs),
                'delta': np.mean(finetuned_accs) - np.mean(baseline_accs)
            })
    return category_data

def compute_subject_deltas(baseline, finetuned):
    """Compute per-subject deltas with metadata."""
    subjects = [k for k in baseline.keys() if k != 'all_subjects']
    data = []
    for s in subjects:
        if s in finetuned:
            data.append({
                'name': s,
                'baseline': baseline[s]['accuracy'],
                'finetuned': finetuned[s]['accuracy'],
                'delta': finetuned[s]['accuracy'] - baseline[s]['accuracy'],
                'examples': baseline[s]['num_examples']
            })
    return sorted(data, key=lambda x: x['delta'], reverse=True)

def plot_category_comparison(ax, category_data):
    """Plot category-level bar chart."""
    categories = [d['category'] for d in category_data]
    baseline_vals = [d['baseline'] * 100 for d in category_data]
    finetuned_vals = [d['finetuned'] * 100 for d in category_data]
    
    y = np.arange(len(categories))
    height = 0.35
    
    bars1 = ax.barh(y + height/2, baseline_vals, height, label='Baseline', color=ORANGE, alpha=0.85)
    bars2 = ax.barh(y - height/2, finetuned_vals, height, label='Fine-tuned', color=BLUE, alpha=0.85)
    
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_xlim(0, 80)
    ax.set_title('Category Performance Comparison')
    ax.legend(loc='lower right', framealpha=1.0)
    ax.grid(axis='x', alpha=0.5, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars1, baseline_vals):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=7, color=ORANGE)
    for bar, val in zip(bars2, finetuned_vals):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=7, color=BLUE)

def plot_delta_scatter(ax, subject_data):
    """Plot scatter of baseline vs finetuned with delta coloring."""
    baselines = [d['baseline'] for d in subject_data]
    finetuneds = [d['finetuned'] for d in subject_data]
    deltas = [d['delta'] for d in subject_data]
    sizes = [np.sqrt(d['examples']) * 3 for d in subject_data]
    
    # Color by delta
    colors = [GREEN if d >= 0 else RED for d in deltas]
    
    scatter = ax.scatter(baselines, finetuneds, c=colors, s=sizes, alpha=0.6, edgecolors='#333333', linewidths=0.5)
    
    # Diagonal line (no change)
    ax.plot([0.2, 0.9], [0.2, 0.9], '--', color=ORANGE, linewidth=2, alpha=0.8, label='No change')
    
    ax.set_xlabel('Baseline Accuracy')
    ax.set_ylabel('Fine-tuned Accuracy')
    ax.set_xlim(0.2, 0.9)
    ax.set_ylim(0.2, 0.9)
    ax.set_title('Performance Delta by Subject')
    ax.grid(alpha=0.5, linestyle='--')
    
    # Format ticks as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markeredgecolor='#333', markersize=10, label='Improved', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED, markeredgecolor='#333', markersize=10, label='Regressed', linestyle='None'),
        Line2D([0], [0], color=ORANGE, linestyle='--', linewidth=2, label='No change'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=1.0, fontsize=8)
    
    # Annotate a few notable points
    notable = sorted(subject_data, key=lambda x: abs(x['delta']), reverse=True)[:5]
    for d in notable:
        short_name = d['name'].replace('high school ', 'HS ').replace('college ', 'C. ')
        if len(short_name) > 15:
            short_name = short_name[:13] + '..'
        ax.annotate(short_name, (d['baseline'], d['finetuned']), 
                   fontsize=6, color='#555555', 
                   xytext=(5, 5), textcoords='offset points')

def main():
    baseline, finetuned = load_data()
    
    category_data = compute_category_stats(baseline, finetuned)
    subject_data = compute_subject_deltas(baseline, finetuned)
    
    # Overall stats
    overall_baseline = baseline['all_subjects']['accuracy']
    overall_finetuned = finetuned['all_subjects']['accuracy']
    overall_delta = overall_finetuned - overall_baseline
    
    # Create figure - single row with 2 plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_category_comparison(axes[0], category_data)
    plot_delta_scatter(axes[1], subject_data)
    
    # Suptitle with overall stats
    fig.suptitle(
        f'MMLU Evaluation: LLaMA 3.1 8B  |  '
        f'Baseline: {overall_baseline*100:.2f}% → Fine-tuned: {overall_finetuned*100:.2f}% (Δ {overall_delta*100:+.2f}%)',
        fontsize=12, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save
    output_path = Path(__file__).parent.parent / 'plots' / 'mmlu_comparison.png'
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_path}")
    
    plt.show()

if __name__ == '__main__':
    main()

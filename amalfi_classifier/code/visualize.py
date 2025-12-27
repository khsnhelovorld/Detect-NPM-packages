#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Import modules (add code directory to path for script execution)
import sys
from pathlib import Path
code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from commons import (
    RESULTS_DIR, VISUALIZATION_DIR, METRICS_FIELD_NAMES
)


def load_evaluation_results(results_file: Optional[Path] = None) -> pd.DataFrame:
    if results_file is None:
        results_file = RESULTS_DIR / 'evaluation_results.csv'
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Evaluation results file not found: {results_file}\n"
            "Please run evaluate.py first to generate evaluation results."
        )
    
    df = pd.read_csv(results_file)
    return df


def plot_metrics_comparison(df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Plot all metrics comparison in a single figure with grouped bar chart.
    All 5 metrics are shown on the same figure for each classifier.
    """
    if output_path is None:
        output_path = VISUALIZATION_DIR / 'metrics_comparison.png'
    
    # Prepare data for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    classifiers = df['classifier'].tolist()
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for grouped bar chart
    x = np.arange(len(classifiers))  # Position of classifiers on x-axis
    width = 0.15  # Width of bars (5 metrics + spacing)
    
    # Color scheme for each metric
    colors = sns.color_palette("husl", len(metrics))
    
    # Plot bars for each metric
    for idx, metric in enumerate(metrics):
        values = df[metric].astype(float).tolist()
        offset = (idx - len(metrics) / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=metric.upper(), 
                     color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classifier performance metrics comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to: {output_path}")


def plot_confusion_matrices(df: pd.DataFrame, output_path: Optional[Path] = None):
    if output_path is None:
        output_path = VISUALIZATION_DIR / 'confusion_matrices.png'
    
    classifiers = df['classifier'].tolist()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion matrices for all classifiers', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, classifier in enumerate(classifiers):
        row = df[df['classifier'] == classifier].iloc[0]
        
        # Build confusion matrix
        cm = [
            [row['true_positive'], row['false_negative']],
            [row['false_positive'], row['true_negative']]
        ]
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Malicious', 'Benign'],
                   yticklabels=['Malicious', 'Benign'],
                   cbar=False)
        axes[idx].set_title(f'{classifier.upper()}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Predicted', fontweight='bold')
        axes[idx].set_ylabel('Actual', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices plot saved to: {output_path}")


def create_all_visualizations(results_file: Optional[Path] = None):
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Load data
    try:
        df = load_evaluation_results(results_file)
        print(f"Loaded evaluation results for {len(df)} classifiers")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    try:
        plot_metrics_comparison(df)
    except Exception as e:
        print(f"Error creating metrics comparison: {e}")
    
    try:
        plot_confusion_matrices(df)
    except Exception as e:
        print(f"Error creating confusion matrices: {e}")
    
    print(f"\n{'='*80}")
    print("ALL VISUALIZATIONS COMPLETED")
    print(f"{'='*80}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        results_file = None
    
    create_all_visualizations(results_file)


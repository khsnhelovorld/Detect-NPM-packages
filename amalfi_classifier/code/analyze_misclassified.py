#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import csv
from typing import List, Dict, Tuple

# Add code directory to path
code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from commons import (
    RESULTS_DIR, VISUALIZATION_DIR, TEST_MALICIOUS_DIR, TEST_NORMAL_DIR
)
from read_features import read_training_features, FEATURE_NAMES


def load_predictions() -> pd.DataFrame:
    predictions_file = RESULTS_DIR / 'detailed_predictions.csv'
    
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\n"
            "Please run export_predictions.py first to generate predictions."
        )
    
    df = pd.read_csv(predictions_file)
    return df


def calculate_misclassification_errors(df: pd.DataFrame) -> pd.DataFrame:
    classifier_columns = [col for col in df.columns if col.startswith('prediction_')]
    
    errors = []
    for idx, row in df.iterrows():
        package_name = row['package_name']
        actual_label = row['actual_label']
        error_count = 0
        error_details = []
        
        for col in classifier_columns:
            predicted_label = row[col]
            if predicted_label != actual_label:
                error_count += 1
                classifier_name = col.replace('prediction_', '')
                error_details.append(classifier_name)
        
        errors.append({
            'package_name': package_name,
            'actual_label': actual_label,
            'error_count': error_count,
            'num_classifiers': len(classifier_columns),
            'errors': ', '.join(error_details) if error_details else 'none'
        })
    
    errors_df = pd.DataFrame(errors)
    return errors_df.sort_values('error_count', ascending=False)


def get_package_features(package_name: str) -> List[float]:
    # Try malicious directory in TEST SET first
    malicious_file = TEST_MALICIOUS_DIR / f"{package_name}.csv"
    if malicious_file.exists():
        from read_features import read_feature_from_csv
        return read_feature_from_csv(str(malicious_file))
    
    # Try normal directory
    normal_file = TEST_NORMAL_DIR / f"{package_name}.csv"
    if normal_file.exists():
        from read_features import read_feature_from_csv
        return read_feature_from_csv(str(normal_file))
    
    return None


def analyze_feature_differences(
    misclassified_packages: List[str],
    all_packages: List[str]
) -> pd.DataFrame:
    misclassified_features = []
    other_features = []
    
    # Get features for misclassified packages
    for pkg in misclassified_packages:
        features = get_package_features(pkg)
        if features:
            misclassified_features.append(features)
    
    # Get features for other packages
    for pkg in all_packages:
        if pkg not in misclassified_packages:
            features = get_package_features(pkg)
            if features:
                other_features.append(features)
    
    if not misclassified_features or not other_features:
        print("Warning: Could not load enough features for comparison")
        return pd.DataFrame()
    
    # Convert to numpy arrays
    misclassified_array = np.array(misclassified_features)
    other_array = np.array(other_features)
    
    # Calculate statistics
    feature_stats = []
    for i, feature_name in enumerate(FEATURE_NAMES):
        misclassified_mean = np.mean(misclassified_array[:, i])
        misclassified_std = np.std(misclassified_array[:, i])
        other_mean = np.mean(other_array[:, i])
        other_std = np.std(other_array[:, i])
        
        # Calculate difference
        diff = misclassified_mean - other_mean
        
        feature_stats.append({
            'feature': feature_name,
            'misclassified_mean': misclassified_mean,
            'misclassified_std': misclassified_std,
            'other_mean': other_mean,
            'other_std': other_std,
            'difference': diff,
            'abs_difference': abs(diff)
        })
    
    stats_df = pd.DataFrame(feature_stats)
    return stats_df.sort_values('abs_difference', ascending=False)


def plot_misclassification_errors(errors_df: pd.DataFrame, top_n: int = 30):
    """Plot top N packages with most misclassification errors."""
    output_path = VISUALIZATION_DIR / 'misclassified_packages.png'
    
    # Get top N packages
    top_packages = errors_df.head(top_n)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top packages by error count
    colors = ['#d62728' if label == 'malicious' else '#2ca02c' 
              for label in top_packages['actual_label']]
    
    ax1.barh(range(len(top_packages)), top_packages['error_count'], color=colors)
    ax1.set_yticks(range(len(top_packages)))
    ax1.set_yticklabels(top_packages['package_name'], fontsize=9)
    ax1.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Packages with Most Misclassifications (Test Set)', 
                  fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Actually Malicious'),
        Patch(facecolor='#2ca02c', label='Actually Benign')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Distribution of error counts
    error_counts = errors_df['error_count'].value_counts().sort_index()
    ax2.bar(error_counts.index, error_counts.values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Packages', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Misclassification Errors (Test Set)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Misclassification plot saved to: {output_path}")
    plt.close()


def plot_feature_differences(stats_df: pd.DataFrame, top_n: int = 15):
    """Plot top N features with largest differences."""
    output_path = VISUALIZATION_DIR / 'feature_differences_misclassified.png'
    
    # Get top N features
    top_features = stats_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grouped bar chart
    x = np.arange(len(top_features))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, top_features['misclassified_mean'], width,
                    label='Misclassified Packages', color='#d62728', alpha=0.8)
    bars2 = ax.barh(x + width/2, top_features['other_mean'], width,
                    label='Other Packages', color='#2ca02c', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Mean Feature Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features with Largest Differences\n(Misclassified vs Other Packages - Test Set)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature differences plot saved to: {output_path}")
    plt.close()


def plot_feature_comparison_table(stats_df: pd.DataFrame, top_n: int = 20):
    """Create a detailed comparison table of features."""
    output_path = VISUALIZATION_DIR / 'feature_comparison_table.png'
    
    # Get top N features
    top_features = stats_df.head(top_n)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in top_features.iterrows():
        table_data.append([
            row['feature'],
            f"{row['misclassified_mean']:.3f} ± {row['misclassified_std']:.3f}",
            f"{row['other_mean']:.3f} ± {row['other_std']:.3f}",
            f"{row['difference']:+.3f}",
            f"{row['abs_difference']:.3f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'Misclassified\n(mean ± std)', 
                              'Other Packages\n(mean ± std)', 
                              'Difference', 'Abs Difference'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.35, 0.18, 0.18, 0.14, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code differences
    for i in range(1, len(table_data) + 1):
        diff = float(table_data[i-1][3])
        if diff > 0:
            table[(i, 3)].set_facecolor('#FFE6E6')  # Light red
        else:
            table[(i, 3)].set_facecolor('#E6F3FF')  # Light blue
    
    plt.title(f'Top {top_n} Features: Misclassified vs Other Packages Comparison (Test Set)',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature comparison table saved to: {output_path}")
    plt.close()


def export_misclassified_analysis(errors_df: pd.DataFrame, stats_df: pd.DataFrame):
    """Export analysis results to CSV files."""
    # Export error analysis
    errors_output = RESULTS_DIR / 'misclassification_errors.csv'
    errors_df.to_csv(errors_output, index=False)
    print(f"Misclassification errors exported to: {errors_output}")
    
    # Export feature statistics
    if not stats_df.empty:
        stats_output = RESULTS_DIR / 'feature_differences_misclassified.csv'
        stats_df.to_csv(stats_output, index=False)
        print(f"Feature differences exported to: {stats_output}")


def main():
    print("="*80)
    print("ANALYZING MISCLASSIFIED PACKAGES (TEST SET ONLY)")
    print("="*80)
    print("\nNOTE: This analysis is performed on TEST SET packages only.")
    print("      Training set packages are NOT included in this analysis.\n")
    
    # Load predictions from test set
    print("Loading predictions from test set...")
    try:
        df = load_predictions()
        print(f"Loaded {len(df)} packages from test set")
        
        # Validate that predictions are from test set only
        # (detailed_predictions.csv should only contain test set packages)
        print(f"  - Test set malicious dir: {TEST_MALICIOUS_DIR}")
        print(f"  - Test set normal dir: {TEST_NORMAL_DIR}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Calculate misclassification errors
    print("\nCalculating misclassification errors...")
    errors_df = calculate_misclassification_errors(df)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("MISCLASSIFICATION SUMMARY (TEST SET)")
    print("="*80)
    print(f"Total test set packages analyzed: {len(errors_df)}")
    print(f"Packages with 0 errors: {len(errors_df[errors_df['error_count'] == 0])}")
    print(f"Packages with 1 error: {len(errors_df[errors_df['error_count'] == 1])}")
    print(f"Packages with 2 errors: {len(errors_df[errors_df['error_count'] == 2])}")
    print(f"Packages with 3 errors: {len(errors_df[errors_df['error_count'] == 3])}")
    print(f"Packages with 4 errors: {len(errors_df[errors_df['error_count'] == 4])}")
    
    # Print top misclassified packages
    print("\n" + "="*80)
    print("TOP 20 PACKAGES WITH MOST MISCLASSIFICATIONS")
    print("="*80)
    top_20 = errors_df.head(20)
    for idx, row in top_20.iterrows():
        label_marker = "[M]" if row['actual_label'] == 'malicious' else "[B]"
        print(f"{label_marker} {row['package_name']:<40} "
              f"Errors: {row['error_count']}/4  ({row['errors']})")
    
    # Plot misclassification errors
    print("\nGenerating misclassification plots...")
    plot_misclassification_errors(errors_df, top_n=30)
    
    # Analyze feature differences for top misclassified packages (test set only)
    print("\nAnalyzing feature differences (test set packages only)...")
    top_misclassified = errors_df.head(50)['package_name'].tolist()
    all_packages = df['package_name'].tolist()  # All packages are from test set
    
    stats_df = analyze_feature_differences(top_misclassified, all_packages)
    
    if not stats_df.empty:
        print("\n" + "="*80)
        print("TOP 10 FEATURES WITH LARGEST DIFFERENCES")
        print("="*80)
        top_10_features = stats_df.head(10)
        for idx, row in top_10_features.iterrows():
            print(f"{row['feature']:<45} "
                  f"Diff: {row['difference']:+.3f} "
                  f"(Misclassified: {row['misclassified_mean']:.3f}, "
                  f"Other: {row['other_mean']:.3f})")
        
        # Plot feature differences
        print("\nGenerating feature difference plots...")
        plot_feature_differences(stats_df, top_n=15)
        plot_feature_comparison_table(stats_df, top_n=20)
    
    # Export results
    print("\nExporting analysis results...")
    export_misclassified_analysis(errors_df, stats_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()


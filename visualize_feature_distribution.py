#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

# Định nghĩa 22 features
FEATURE_NAMES = [
    'hasInstallScript',
    'containIP',
    'useBase64Conversion',
    'useBase64ConversionInInstallScript',
    'containBase64StringInJSFile',
    'containBase64StringInInstallScript',
    'containBytestring',
    'containDomainInJSFile',
    'containDomainInInstallScript',
    'useBuffer',
    'useEval',
    'requireChildProcessInJSFile',
    'requireChildProcessInInstallScript',
    'accessFSInJSFile',
    'accessFSInInstallScript',
    'accessNetworkInJSFile',
    'accessNetworkInInstallScript',
    'accessProcessEnvInJSFile',
    'accessProcessEnvInInstallScript',
    'containSuspicousString',
    'accessCryptoAndZip',
    'accessSensitiveAPI'
]

def parse_percentage(value_str):
    """Parse percentage string like '88.26%' to float"""
    return float(value_str.replace('%', ''))

def load_statistics_from_csv(csv_file):
    """Load statistics from CSV file"""
    stats = {
        'train_mal': {},
        'train_ben': {},
        'test_mal': {},
        'test_ben': {},
        'combined_mal': {},
        'combined_ben': {}
    }
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = row['Feature']
            stats['train_mal'][feature] = parse_percentage(row['Train_Mal_%'])
            stats['train_ben'][feature] = parse_percentage(row['Train_Ben_%'])
            stats['test_mal'][feature] = parse_percentage(row['Test_Mal_%'])
            stats['test_ben'][feature] = parse_percentage(row['Test_Ben_%'])
            stats['combined_mal'][feature] = parse_percentage(row['Combined_Mal_%'])
            stats['combined_ben'][feature] = parse_percentage(row['Combined_Ben_%'])
    
    return stats

def plot_comparison_bar_chart(stats, dataset_type='combined', save_path=None):
    """Vẽ biểu đồ cột so sánh tỉ lệ features giữa malicious và benign"""
    mal_key = f'{dataset_type}_mal'
    ben_key = f'{dataset_type}_ben'
    
    features = FEATURE_NAMES
    mal_percentages = [stats[mal_key][f] for f in features]
    ben_percentages = [stats[ben_key][f] for f in features]
    
    # Shorten feature names for better display
    short_features = [f.replace('InInstallScript', 'InInst').replace('InJSFile', 'InJS')
                     .replace('require', 'req').replace('access', 'acc')
                     .replace('contain', 'cont').replace('Conversion', 'Conv')
                     for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(20, 10))
    bars1 = ax.bar(x - width/2, mal_percentages, width, label='Malicious', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, ben_percentages, width, label='Benign', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Distribution Comparison - {dataset_type.upper()} Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(short_features, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Only label bars that are significant (>5%)
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if mal_percentages[i] > 5:
            ax.text(bar1.get_x() + bar1.get_width()/2., mal_percentages[i],
                   f'{mal_percentages[i]:.1f}%',
                   ha='center', va='bottom', fontsize=7, rotation=90)
        if ben_percentages[i] > 5:
            ax.text(bar2.get_x() + bar2.get_width()/2., ben_percentages[i],
                   f'{ben_percentages[i]:.1f}%',
                   ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_difference_chart(stats, dataset_type='combined', save_path=None):
    """Vẽ biểu đồ hiển thị sự khác biệt giữa malicious và benign"""
    mal_key = f'{dataset_type}_mal'
    ben_key = f'{dataset_type}_ben'
    
    features = FEATURE_NAMES
    differences = [stats[mal_key][f] - stats[ben_key][f] for f in features]
    
    # Shorten feature names
    short_features = [f.replace('InInstallScript', 'InInst').replace('InJSFile', 'InJS')
                     .replace('require', 'req').replace('access', 'acc')
                     .replace('contain', 'cont').replace('Conversion', 'Conv')
                     for f in features]
    
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in differences]
    
    fig, ax = plt.subplots(figsize=(20, 8))
    bars = ax.barh(range(len(features)), differences, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(short_features, fontsize=9)
    ax.set_xlabel('Difference (Malicious % - Benign %)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Difference: Malicious vs Benign - {dataset_type.upper()} Dataset',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        if abs(diff) > 1:
            ax.text(diff, i, f'{diff:+.1f}%',
                   va='center', ha='left' if diff > 0 else 'right',
                   fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_heatmap(stats, dataset_type='combined', save_path=None):
    """Vẽ heatmap so sánh features"""
    mal_key = f'{dataset_type}_mal'
    ben_key = f'{dataset_type}_ben'
    
    features = FEATURE_NAMES
    data = np.array([
        [stats[mal_key][f] for f in features],
        [stats[ben_key][f] for f in features]
    ])
    
    # Shorten feature names
    short_features = [f.replace('InInstallScript', 'InInst').replace('InJSFile', 'InJS')
                     .replace('require', 'req').replace('access', 'acc')
                     .replace('contain', 'cont').replace('Conversion', 'Conv')
                     for f in features]
    
    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(short_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Malicious', 'Benign'], fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Distribution Heatmap - {dataset_type.upper()} Dataset',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add text annotations
    for i in range(2):
        for j in range(len(features)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha="center", va="center", color="black" if data[i, j] < 50 else "white",
                          fontsize=7, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_top_features(stats, dataset_type='combined', top_n=10, save_path=None):
    """Vẽ biểu đồ top N features có sự khác biệt lớn nhất"""
    mal_key = f'{dataset_type}_mal'
    ben_key = f'{dataset_type}_ben'
    
    features = FEATURE_NAMES
    differences = [(f, stats[mal_key][f] - stats[ben_key][f]) for f in features]
    differences.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = differences[:top_n]
    
    feature_names = [f[0].replace('InInstallScript', 'InInst').replace('InJSFile', 'InJS')
                    .replace('require', 'req').replace('access', 'acc')
                    .replace('contain', 'cont').replace('Conversion', 'Conv')
                    for f in top_features]
    diff_values = [f[1] for f in top_features]
    
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff_values]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(feature_names)), diff_values, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_xlabel('Difference (Malicious % - Benign %)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features with Largest Difference - {dataset_type.upper()} Dataset',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, diff_values)):
        ax.text(diff, i, f'{diff:+.1f}%',
               va='center', ha='left' if diff > 0 else 'right',
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_all_datasets_comparison(stats, save_path=None):
    """Vẽ biểu đồ so sánh giữa Training, Test và Combined datasets"""
    datasets = ['train', 'test', 'combined']
    dataset_labels = ['Training', 'Test', 'Combined']
    
    # Select top 10 features with largest difference in combined dataset
    features = FEATURE_NAMES
    differences = [(f, stats['combined_mal'][f] - stats['combined_ben'][f]) for f in features]
    differences.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [f[0] for f in differences[:10]]
    
    # Shorten feature names
    short_features = [f.replace('InInstallScript', 'InInst').replace('InJSFile', 'InJS')
                     .replace('require', 'req').replace('access', 'acc')
                     .replace('contain', 'cont').replace('Conversion', 'Conv')
                     for f in top_features]
    
    x = np.arange(len(top_features))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    for i, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        mal_key = f'{dataset}_mal'
        ben_key = f'{dataset}_ben'
        mal_values = [stats[mal_key][f] for f in top_features]
        ben_values = [stats[ben_key][f] for f in top_features]
        
        offset = (i - 1) * width
        ax.bar(x + offset, mal_values, width, label=f'{label} - Malicious', alpha=0.8)
        ax.bar(x + offset + width/2, ben_values, width, label=f'{label} - Benign', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Features Comparison Across All Datasets', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(short_features, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def main():
    # Đường dẫn đến file CSV statistics
    script_dir = Path(__file__).parent.absolute()
    csv_file = script_dir / 'feature_statistics.csv'
    
    if not csv_file.exists():
        print(f"ERROR: Statistics file not found at: {csv_file}")
        print("Please run Features_dataset_analysis.py first to generate the statistics file.")
        return
    
    print("="*100)
    print("FEATURE DISTRIBUTION VISUALIZATION")
    print("="*100)
    print(f"Loading statistics from: {csv_file}")
    
    # Load statistics
    stats = load_statistics_from_csv(csv_file)
    
    output_dir = script_dir / 'feature_plots'
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # 1. Comparison bar chart for combined dataset
    print("\n1. Creating comparison bar chart (Combined)...")
    plot_comparison_bar_chart(stats, dataset_type='combined', 
                             save_path=output_dir / 'comparison_combined.png')
    
    # 2. Comparison bar chart for training dataset
    print("\n2. Creating comparison bar chart (Training)...")
    plot_comparison_bar_chart(stats, dataset_type='train',
                             save_path=output_dir / 'comparison_training.png')
    
    # 3. Comparison bar chart for test dataset
    print("\n3. Creating comparison bar chart (Test)...")
    plot_comparison_bar_chart(stats, dataset_type='test',
                             save_path=output_dir / 'comparison_test.png')
    
    # 4. Difference chart for combined dataset
    print("\n4. Creating difference chart (Combined)...")
    plot_difference_chart(stats, dataset_type='combined',
                          save_path=output_dir / 'difference_combined.png')
    
    # 5. Heatmap for combined dataset
    print("\n5. Creating heatmap (Combined)...")
    plot_heatmap(stats, dataset_type='combined',
                save_path=output_dir / 'heatmap_combined.png')
    
    # 6. Top 10 features with largest difference
    print("\n6. Creating top 10 features chart (Combined)...")
    plot_top_features(stats, dataset_type='combined', top_n=10,
                     save_path=output_dir / 'top10_features_combined.png')
    
    # 7. All datasets comparison
    print("\n7. Creating all datasets comparison chart...")
    plot_all_datasets_comparison(stats,
                                save_path=output_dir / 'all_datasets_comparison.png')
    
    print("\n" + "="*100)
    print("All visualizations completed!")
    print(f"Plots saved to: {output_dir}")
    print("="*100)

if __name__ == '__main__':
    main()


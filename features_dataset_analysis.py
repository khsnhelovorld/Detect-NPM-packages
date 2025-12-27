#!/usr/bin/env python3
import csv
import os
from collections import defaultdict
from pathlib import Path

# Định nghĩa 22 features theo thứ tự
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

def read_csv_directory(directory_path):
    #Đọc tất cả CSV files trong directory và return feature vectors
    feature_data = []
    csv_files = list(Path(directory_path).glob('*.csv'))
    
    print(f"Reading {len(csv_files)} CSV files from {directory_path}")
    
    for csv_file in csv_files:
        try:
            features = {}
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        feature_name = row[0].strip()
                        feature_value = row[1].strip().lower() == 'true'
                        features[feature_name] = feature_value
            
            # Convert to list in correct order
            feature_vector = [features.get(name, False) for name in FEATURE_NAMES]
            feature_data.append(feature_vector)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    return feature_data

def calculate_feature_statistics(feature_data, label_name):
    #Tính toán thống kê cho features
    if not feature_data:
        return {}
    
    num_samples = len(feature_data)
    num_features = len(FEATURE_NAMES)
    
    # Đếm số lượng samples có feature = true cho mỗi feature
    feature_counts = [0] * num_features
    for sample in feature_data:
        for i, feature_value in enumerate(sample):
            if feature_value:
                feature_counts[i] += 1
    
    # Tính tỉ lệ
    feature_ratios = [count / num_samples for count in feature_counts]
    
    stats = {}
    for i, feature_name in enumerate(FEATURE_NAMES):
        stats[feature_name] = {
            'count': feature_counts[i],
            'ratio': feature_ratios[i],
            'percentage': feature_ratios[i] * 100
        }
    
    return stats, num_samples

def compare_features(malicious_stats, benign_stats, malicious_count, benign_count):
    #So sánh tỉ lệ features giữa malicious và benign
    print("\n" + "="*100)
    print("FEATURE DISTRIBUTION COMPARISON")
    print("="*100)
    print(f"{'Feature Name':<40} {'Malicious %':<15} {'Benign %':<15} {'Difference':<15} {'Ratio':<15}")
    print("-"*100)
    
    for feature_name in FEATURE_NAMES:
        mal_ratio = malicious_stats[feature_name]['percentage']
        ben_ratio = benign_stats[feature_name]['percentage']
        diff = mal_ratio - ben_ratio
        ratio = mal_ratio / ben_ratio if ben_ratio > 0 else float('inf')
        
        print(f"{feature_name:<40} {mal_ratio:>6.2f}% ({malicious_stats[feature_name]['count']:>5}/{malicious_count:<5}) "
              f"{ben_ratio:>6.2f}% ({benign_stats[feature_name]['count']:>5}/{benign_count:<5}) "
              f"{diff:>+6.2f}% {ratio:>6.2f}x")

def main():
    script_dir = Path(__file__).parent.absolute()  # Workspace/
    base_dir = script_dir / 'dataset'  # Workspace/dataset
    
    train_malicious_dir = base_dir / 'training_set' / 'malicious'
    train_benign_dir = base_dir / 'training_set' / 'normal'
    test_malicious_dir = base_dir / 'test_set' / 'malicious-dedupli'
    test_benign_dir = base_dir / 'test_set' / 'normal'
    
    if not base_dir.exists():
        print(f"ERROR: Dataset directory not found at: {base_dir}")
        print(f"Current script location: {script_dir}")
        return
    
    print("="*100)
    print("FEATURE EXTRACTION ANALYSIS")
    print("="*100)
    print(f"Dataset base directory: {base_dir}")
    print(f"Output file will be saved to: {script_dir / 'feature_statistics.csv'}")
    
    print("\n--- TRAINING SET ---")
    train_malicious_data = read_csv_directory(train_malicious_dir)
    train_benign_data = read_csv_directory(train_benign_dir)
    
    train_malicious_stats, train_mal_count = calculate_feature_statistics(
        train_malicious_data, 'Training Malicious'
    )
    train_benign_stats, train_ben_count = calculate_feature_statistics(
        train_benign_data, 'Training Benign'
    )
    
    print(f"\nTraining Set - Malicious: {train_mal_count} samples")
    print(f"Training Set - Benign: {train_ben_count} samples")
    
    compare_features(train_malicious_stats, train_benign_stats, train_mal_count, train_ben_count)
    
    print("\n\n--- TEST SET ---")
    test_malicious_data = read_csv_directory(test_malicious_dir)
    test_benign_data = read_csv_directory(test_benign_dir)
    
    test_malicious_stats, test_mal_count = calculate_feature_statistics(
        test_malicious_data, 'Test Malicious'
    )
    test_benign_stats, test_ben_count = calculate_feature_statistics(
        test_benign_data, 'Test Benign'
    )
    
    print(f"\nTest Set - Malicious: {test_mal_count} samples")
    print(f"Test Set - Benign: {test_ben_count} samples")
    
    compare_features(test_malicious_stats, test_benign_stats, test_mal_count, test_ben_count)
    
    # Combined statistics
    print("\n\n--- COMBINED (TRAINING + TEST) ---")
    combined_malicious_data = train_malicious_data + test_malicious_data
    combined_benign_data = train_benign_data + test_benign_data
    
    combined_malicious_stats, combined_mal_count = calculate_feature_statistics(
        combined_malicious_data, 'Combined Malicious'
    )
    combined_benign_stats, combined_ben_count = calculate_feature_statistics(
        combined_benign_data, 'Combined Benign'
    )
    
    print(f"\nCombined - Malicious: {combined_mal_count} samples")
    print(f"Combined - Benign: {combined_ben_count} samples")
    
    compare_features(combined_malicious_stats, combined_benign_stats, 
                     combined_mal_count, combined_ben_count)
    
    # Export to CSV
    print("\n\n--- EXPORTING STATISTICS TO CSV ---")
    output_file = script_dir / 'feature_statistics.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Train_Mal_Count', 'Train_Mal_%', 
                        'Train_Ben_Count', 'Train_Ben_%',
                        'Test_Mal_Count', 'Test_Mal_%',
                        'Test_Ben_Count', 'Test_Ben_%',
                        'Combined_Mal_Count', 'Combined_Mal_%',
                        'Combined_Ben_Count', 'Combined_Ben_%'])
        
        for feature_name in FEATURE_NAMES:
            writer.writerow([
                feature_name,
                train_malicious_stats[feature_name]['count'],
                f"{train_malicious_stats[feature_name]['percentage']:.2f}%",
                train_benign_stats[feature_name]['count'],
                f"{train_benign_stats[feature_name]['percentage']:.2f}%",
                test_malicious_stats[feature_name]['count'],
                f"{test_malicious_stats[feature_name]['percentage']:.2f}%",
                test_benign_stats[feature_name]['count'],
                f"{test_benign_stats[feature_name]['percentage']:.2f}%",
                combined_malicious_stats[feature_name]['count'],
                f"{combined_malicious_stats[feature_name]['percentage']:.2f}%",
                combined_benign_stats[feature_name]['count'],
                f"{combined_benign_stats[feature_name]['percentage']:.2f}%"
            ])
    
    print(f"Statistics exported to {output_file}")
    print(f"Full path: {output_file.absolute()}")

if __name__ == '__main__':
    main()
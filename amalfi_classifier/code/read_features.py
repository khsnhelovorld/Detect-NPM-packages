#!/usr/bin/env python3
# Đọc features từ CSV files

import csv
import os
from pathlib import Path
from typing import List, Tuple


# 22 features theo thứ tự như trong dataset
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


def read_feature_from_csv(csv_path: str) -> List[float]:
    features_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows or rows with insufficient columns
            if len(row) >= 2:
                feature_name = row[0].strip()
                feature_value = row[1].strip().lower()
                
                # Convert boolean string to float
                if feature_value == 'true':
                    features_dict[feature_name] = 1.0
                elif feature_value == 'false':
                    features_dict[feature_name] = 0.0
                else:
                    # Handle unexpected values (default to 0.0)
                    features_dict[feature_name] = 0.0
    
    # Convert dictionary to list in the correct feature order
    # Missing features default to 0.0
    feature_vector = [features_dict.get(name, 0.0) for name in FEATURE_NAMES]
    
    return feature_vector


def read_features_from_directory(directory_path: str, is_malicious: bool) -> Tuple[List[List[float]], List[str], List[str]]:
    feature_vectors = []
    labels = []
    csv_filenames = []
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Warning: Directory {directory_path} does not exist")
        return feature_vectors, labels, csv_filenames
    
    # Find all CSV files in the directory
    csv_files = list(directory.glob('*.csv'))
    
    print(f"Reading {len(csv_files)} CSV files from {directory_path}")
    
    for csv_file in csv_files:
        try:
            feature_vec = read_feature_from_csv(str(csv_file))
            feature_vectors.append(feature_vec)
            labels.append("malicious" if is_malicious else "benign")
            csv_filenames.append(csv_file.name)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    return feature_vectors, labels, csv_filenames


def read_training_features(malicious_dir: str, normal_dir: str) -> Tuple[List[List[float]], List[str], List[str]]:
    all_feature_vectors = []
    all_labels = []
    all_filenames = []
    
    # Read malicious samples
    if malicious_dir:
        mal_features, mal_labels, mal_filenames = read_features_from_directory(
            malicious_dir, is_malicious=True
        )
        all_feature_vectors.extend(mal_features)
        all_labels.extend(mal_labels)
        all_filenames.extend(mal_filenames)
    
    # Read normal samples
    if normal_dir:
        norm_features, norm_labels, norm_filenames = read_features_from_directory(
            normal_dir, is_malicious=False
        )
        all_feature_vectors.extend(norm_features)
        all_labels.extend(norm_labels)
        all_filenames.extend(norm_filenames)
    
    # Print summary statistics
    malicious_count = sum(1 for l in all_labels if l == 'malicious')
    benign_count = sum(1 for l in all_labels if l == 'benign')
    print(f"Total samples loaded: {len(all_feature_vectors)} "
          f"({malicious_count} malicious, {benign_count} benign)")
    
    return all_feature_vectors, all_labels, all_filenames


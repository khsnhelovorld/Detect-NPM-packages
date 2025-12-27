#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path

code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from read_features import read_feature_from_csv, FEATURE_NAMES
from commons import (
    DECISION_TREE_PATH, NAIVE_BAYES_PATH, SVM_PATH, RANDOM_FOREST_PATH,
    CLASSIFIER_DIR
)

def load_classifier(classifier_path: Path):
    with open(classifier_path, 'rb') as f:
        classifier_data = pickle.load(f)
    return classifier_data

def predict_single_package(csv_path: str, classifier_data: dict) -> str:
    # Read features
    feature_vec = read_feature_from_csv(csv_path)
    
    # Get classifier and feature names
    clf = classifier_data['classifier']
    excluded_features = classifier_data.get('excluded_features', [])
    all_feature_names = classifier_data.get('all_feature_names', FEATURE_NAMES)
    
    # Filter out excluded features
    if excluded_features:
        exclude_indices = [all_feature_names.index(f) for f in excluded_features if f in all_feature_names]
        feature_vec = [v for i, v in enumerate(feature_vec) if i not in exclude_indices]
    
    # Apply booleanization if needed
    booleanize = classifier_data.get('booleanize', False)
    if booleanize:
        feature_vec = [1.0 if v > 0 else 0.0 for v in feature_vec]
    
    # Apply positive filter if needed
    positive = classifier_data.get('positive', False)
    if positive:
        feature_vec = [max(0.0, v) for v in feature_vec]
    
    # Predict
    # SVC returns labels directly (no conversion needed)
    prediction = clf.predict([feature_vec])[0]
    return prediction

def predict_directory(csv_directory: str, classifier_path: Path, output_path: Path = None):
    # Load classifier
    print(f"Loading classifier from {classifier_path}...")
    classifier_data = load_classifier(classifier_path)
    classifier_type = classifier_data.get('classifier_type', 'unknown')
    print(f"Classifier type: {classifier_type}")
    
    # Find all CSV files
    csv_dir = Path(csv_directory)
    csv_files = list(csv_dir.glob('*.csv'))
    
    print(f"Predicting {len(csv_files)} packages...")
    
    results = []
    for csv_file in csv_files:
        try:
            prediction = predict_single_package(str(csv_file), classifier_data)
            results.append({
                'package': csv_file.stem,
                'prediction': prediction,
                'file': csv_file.name
            })
            print(f"{csv_file.name}: {prediction}")
        except Exception as e:
            print(f"Error predicting {csv_file.name}: {e}")
            results.append({
                'package': csv_file.stem,
                'prediction': 'error',
                'file': csv_file.name
            })
    
    # Save results if output path provided
    if output_path:
        import csv as csv_module
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv_module.DictWriter(f, fieldnames=['package', 'prediction', 'file'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {output_file}")
    
    # Summary
    malicious_count = sum(1 for r in results if r['prediction'] == 'malicious')
    benign_count = sum(1 for r in results if r['prediction'] == 'benign')
    
    print(f"\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  Predicted malicious: {malicious_count}")
    print(f"  Predicted benign: {benign_count}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict packages using trained classifier")
    parser.add_argument(
        'classifier',
        choices=['decision-tree', 'naive-bayes', 'svm', 'random-forest'],
        help='Classifier to use'
    )
    parser.add_argument(
        'csv_path',
        help='Path to CSV file or directory containing CSV files'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file to save predictions (optional)'
    )
    parser.add_argument(
        '--classifier-file',
        help='Custom classifier file path (default: from commons.py)'
    )
    
    args = parser.parse_args()
    
    # Get classifier path
    classifier_paths = {
        'decision-tree': DECISION_TREE_PATH,
        'naive-bayes': NAIVE_BAYES_PATH,
        'svm': SVM_PATH,
        'random-forest': RANDOM_FOREST_PATH
    }
    
    classifier_path = Path(args.classifier_file) if args.classifier_file else classifier_paths[args.classifier]
    
    if not classifier_path.exists():
        print(f"Error: Classifier file not found: {classifier_path}")
        print("Please train the classifier first using train.py")
        sys.exit(1)
    
    csv_path = Path(args.csv_path)
    
    if csv_path.is_file():
        # Single file prediction
        classifier_data = load_classifier(classifier_path)
        prediction = predict_single_package(str(csv_path), classifier_data)
        print(f"Prediction: {prediction}")
    elif csv_path.is_dir():
        # Directory prediction
        predict_directory(str(csv_path), classifier_path, args.output)
    else:
        print(f"Error: {csv_path} is not a valid file or directory")
        sys.exit(1)


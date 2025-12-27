#!/usr/bin/env python3
import pickle
import csv
from pathlib import Path
import sys

code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from read_features import read_training_features
from commons import (
    DECISION_TREE_PATH, NAIVE_BAYES_PATH, SVM_PATH, RANDOM_FOREST_PATH,
    TEST_MALICIOUS_DIR, TEST_NORMAL_DIR, RESULTS_DIR
)

def load_classifier(classifier_path: Path):
    with open(classifier_path, 'rb') as f:
        return pickle.load(f)

def predict_with_classifier(feature_vectors, classifier_data):
    clf = classifier_data['classifier']
    excluded_features = classifier_data.get('excluded_features', [])
    all_feature_names = classifier_data.get('all_feature_names', [])
    booleanize = classifier_data.get('booleanize', False)
    positive = classifier_data.get('positive', False)
    
    # Process feature vectors
    processed_vectors = []
    for vec in feature_vectors:
        # Filter excluded features
        if excluded_features:
            exclude_indices = [all_feature_names.index(f) for f in excluded_features if f in all_feature_names]
            vec = [v for i, v in enumerate(vec) if i not in exclude_indices]
        
        # Apply booleanization
        if booleanize:
            vec = [1.0 if v > 0 else 0.0 for v in vec]
        
        # Apply positive filter
        if positive:
            vec = [max(0.0, v) for v in vec]
        
        processed_vectors.append(vec)
    
    # Predict
    predictions = clf.predict(processed_vectors)
    return predictions

def get_package_name_from_filename(filename: str) -> str:
    """Extract package name from CSV filename (removes .csv extension)"""
    return Path(filename).stem

def main():
    print("="*80)
    print("EXPORTING DETAILED PREDICTIONS FOR ALL CLASSIFIERS")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    test_features, test_labels, test_filenames = read_training_features(
        str(TEST_MALICIOUS_DIR), str(TEST_NORMAL_DIR)
    )
    
    if len(test_features) == 0:
        print("Error: No test data loaded")
        return
    
    print(f"Loaded {len(test_features)} test samples")
    
    # Classifier definitions
    classifiers = [
        ("decision-tree", DECISION_TREE_PATH),
        ("naive-bayes", NAIVE_BAYES_PATH),
        ("svm", SVM_PATH),
        ("random-forest", RANDOM_FOREST_PATH),
    ]
    
    # Store predictions for each classifier
    all_predictions = {}
    
    # Get predictions from each classifier
    for classifier_name, classifier_path in classifiers:
        if not classifier_path.exists():
            print(f"\nWarning: {classifier_name} classifier not found. Skipping.")
            continue
        
        print(f"\nPredicting with {classifier_name}...")
        try:
            classifier_data = load_classifier(classifier_path)
            predictions = predict_with_classifier(test_features, classifier_data)
            all_predictions[classifier_name] = predictions
            print(f"  Completed: {len(predictions)} predictions")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Export combined CSV (all classifiers in one file)
    output_file = RESULTS_DIR / 'detailed_predictions.csv'
    print(f"\n{'='*80}")
    print(f"Exporting combined predictions to: {output_file}")
    print(f"{'='*80}")
    
    # Prepare column names
    fieldnames = ['package_name', 'actual_label']
    for classifier_name, _ in classifiers:
        if classifier_name in all_predictions:
            fieldnames.append(f'prediction_{classifier_name}')
    
    # Write combined CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, filename in enumerate(test_filenames):
            package_name = get_package_name_from_filename(filename)
            row = {
                'package_name': package_name,
                'actual_label': test_labels[idx]
            }
            
            # Add predictions from each classifier
            for classifier_name in all_predictions.keys():
                row[f'prediction_{classifier_name}'] = all_predictions[classifier_name][idx]
            
            writer.writerow(row)
    
    print(f"Combined predictions exported successfully!")
    
    # Export separate CSV for each classifier (malicious packages only)
    print(f"\n{'='*80}")
    print("Exporting malicious packages per classifier:")
    print(f"{'='*80}")
    
    malicious_packages_per_classifier = {}
    
    for classifier_name in all_predictions.keys():
        malicious_packages = []
        
        for idx, filename in enumerate(test_filenames):
            package_name = get_package_name_from_filename(filename)
            if all_predictions[classifier_name][idx] == 'malicious':
                malicious_packages.append({
                    'package_name': package_name,
                    'actual_label': test_labels[idx]
                })
        
        malicious_packages_per_classifier[classifier_name] = malicious_packages
        
        # Write CSV for this classifier
        classifier_output = RESULTS_DIR / f'malicious_packages_{classifier_name}.csv'
        with open(classifier_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['package_name', 'actual_label'])
            writer.writeheader()
            writer.writerows(malicious_packages)
        
        print(f"  {classifier_name:<20}: {len(malicious_packages):4d} packages -> {classifier_output.name}")
    
    print(f"\nTotal packages in test set: {len(test_filenames)}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY: Packages predicted as MALICIOUS by each classifier")
    print(f"{'='*80}")
    
    for classifier_name in all_predictions.keys():
        malicious_count = sum(1 for p in all_predictions[classifier_name] if p == 'malicious')
        print(f"{classifier_name:<20}: {malicious_count:4d} packages ({malicious_count/len(test_filenames)*100:.1f}%)")
    
    # Print packages predicted as malicious by all classifiers
    print(f"\n{'='*80}")
    print("Packages predicted as MALICIOUS by ALL classifiers:")
    print(f"{'='*80}")
    
    all_malicious = []
    for idx, filename in enumerate(test_filenames):
        package_name = get_package_name_from_filename(filename)
        # Check if all classifiers predict malicious
        if all(all_predictions.get(cn, [])[idx] == 'malicious' 
               for cn in all_predictions.keys()):
            all_malicious.append(package_name)
    
    if all_malicious:
        for pkg in sorted(all_malicious):
            print(f"  - {pkg}")
        print(f"\nTotal: {len(all_malicious)} packages")
    else:
        print("  (None)")
    
    # Print packages predicted as malicious by at least one classifier
    print(f"\n{'='*80}")
    print("Packages predicted as MALICIOUS by AT LEAST ONE classifier:")
    print(f"{'='*80}")
    
    at_least_one_malicious = []
    for idx, filename in enumerate(test_filenames):
        package_name = get_package_name_from_filename(filename)
        # Check if at least one classifier predicts malicious
        if any(all_predictions.get(cn, [])[idx] == 'malicious' 
               for cn in all_predictions.keys()):
            at_least_one_malicious.append(package_name)
    
    print(f"Total: {len(at_least_one_malicious)} packages")
    if len(at_least_one_malicious) <= 50:
        for pkg in sorted(at_least_one_malicious):
            print(f"  - {pkg}")
    else:
        print(f"  (First 50 packages shown)")
        for pkg in sorted(at_least_one_malicious)[:50]:
            print(f"  - {pkg}")
        print(f"  ... and {len(at_least_one_malicious) - 50} more")

if __name__ == '__main__':
    main()
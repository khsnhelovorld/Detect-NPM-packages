#!/usr/bin/env python3
import pickle
import csv
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import sys
from pathlib import Path
code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from read_features import read_training_features
from commons import (
    DECISION_TREE_PATH, NAIVE_BAYES_PATH, SVM_PATH, RANDOM_FOREST_PATH,
    TEST_MALICIOUS_DIR, TEST_NORMAL_DIR, RESULTS_DIR,
    METRICS_FIELD_NAMES
)

def load_classifier(classifier_path: Path):
    with open(classifier_path, 'rb') as f:
        return pickle.load(f)

def predict_with_classifier(feature_vectors, labels, classifier_data):
    clf = classifier_data['classifier']
    classifier_type = classifier_data.get('classifier_type', 'unknown')
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
    # SVC returns labels directly (no conversion needed)
    predictions = clf.predict(processed_vectors)
    
    return predictions

def evaluate_classifier(classifier_path: Path, test_malicious_dir: str, test_normal_dir: str):
    print(f"\n{'='*80}")
    print(f"Evaluating classifier: {classifier_path.name}")
    print(f"{'='*80}")
    
    # Load classifier
    classifier_data = load_classifier(classifier_path)
    classifier_type = classifier_data.get('classifier_type', 'unknown')
    print(f"Classifier type: {classifier_type}")
    
    # Load test data
    print("Loading test data...")
    test_features, test_labels, test_filenames = read_training_features(test_malicious_dir, test_normal_dir)
    
    if len(test_features) == 0:
        print("Error: No test data loaded")
        return None
    
    print(f"Test samples: {len(test_features)} ({sum(1 for l in test_labels if l == 'malicious')} malicious, {sum(1 for l in test_labels if l == 'benign')} benign)")
    
    # Predict
    print("Predicting...")
    predictions = predict_with_classifier(test_features, test_labels, classifier_data)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, pos_label="malicious", zero_division=0)
    recall = recall_score(test_labels, predictions, pos_label="malicious", zero_division=0)
    f1 = f1_score(test_labels, predictions, pos_label="malicious", zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions, labels=["malicious", "benign"])
    # cm format: [[TP, FN], [FP, TN]] for labels=["malicious", "benign"]
    tp = cm[0][0]  # True Positive: malicious predicted as malicious
    fn = cm[0][1]  # False Negative: malicious predicted as benign
    fp = cm[1][0]  # False Positive: benign predicted as malicious
    tn = cm[1][1]  # True Negative: benign predicted as benign
    
    # MCC (Matthews Correlation Coefficient)
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    results = {
        'classifier': classifier_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(
        test_labels, 
        predictions, 
        target_names=['benign', 'malicious'],
        digits=2,
        zero_division=0
    ))
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Malicious  Benign")
    print(f"  Malicious      {tp:4d}    {fn:4d}")
    print(f"  Benign         {fp:4d}    {tn:4d}")
    
    # Print TP, FP, TN, FN table
    print(f"\nDetailed Metrics Table:")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'True Positive (TP)':<20} {tp:<10}")
    print(f"{'True Negative (TN)':<20} {tn:<10}")
    print(f"{'False Positive (FP)':<20} {fp:<10}")
    print(f"{'False Negative (FN)':<20} {fn:<10}")
    print(f"{'Total Samples':<20} {tp+tn+fp+fn:<10}")
    
    # Print summary metrics
    print(f"\nSummary Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    
    return results

def main():
    print("="*80)
    print("EVALUATING CLASSIFIERS ON TEST SET")
    print("="*80)
    print(f"Test data:")
    print(f"  Malicious: {TEST_MALICIOUS_DIR}")
    print(f"  Normal:    {TEST_NORMAL_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("="*80)
    
    classifiers = [
        ("Decision Tree", DECISION_TREE_PATH),
        ("Naive Bayes", NAIVE_BAYES_PATH),
        ("SVM", SVM_PATH),
        ("Random Forest", RANDOM_FOREST_PATH),
    ]
    
    all_results = []
    
    for classifier_name, classifier_path in classifiers:
        if not classifier_path.exists():
            print(f"\nWarning: {classifier_name} classifier not found at {classifier_path}")
            print("Skipping. Please train it first using train.py")
            continue
        
        try:
            results = evaluate_classifier(classifier_path, str(TEST_MALICIOUS_DIR), str(TEST_NORMAL_DIR))
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"\nError evaluating {classifier_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    if all_results:
        results_file = RESULTS_DIR / 'evaluation_results.csv'
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_FIELD_NAMES + ['true_positive', 'true_negative', 'false_positive', 'false_negative'])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    'classifier': r['classifier'],
                    'accuracy': f"{r['accuracy']:.4f}",
                    'precision': f"{r['precision']:.4f}",
                    'recall': f"{r['recall']:.4f}",
                    'f1': f"{r['f1']:.4f}",
                    'mcc': f"{r['mcc']:.4f}",
                    'true_positive': r['true_positive'],
                    'true_negative': r['true_negative'],
                    'false_positive': r['false_positive'],
                    'false_negative': r['false_negative']
                })
        
        print(f"\n{'='*80}")
        print("ALL EVALUATIONS COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved to: {results_file}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(f"{'Classifier':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
        print("-"*80)
        for r in all_results:
            print(f"{r['classifier']:<20} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['mcc']:<10.4f}")
    else:
        print("\nNo classifiers were evaluated successfully.")

if __name__ == '__main__':
    main()


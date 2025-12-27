#!/usr/bin/env python3
#This script trains all classifiers (Decision Tree, Naive Bayes, SVM, Random Forest) on the dataset. It serves as the main entry point for classifier training.

import sys
from pathlib import Path
code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from train_classifier import train_classifier
from commons import (
    TRAINING_MALICIOUS_DIR, TRAINING_NORMAL_DIR,
    DECISION_TREE_PATH, NAIVE_BAYES_PATH, SVM_PATH, RANDOM_FOREST_PATH,
    RESULTS_DIR
)


def main():
    print("="*80)
    print("AMALFI-STYLE CLASSIFIER TRAINING")
    print("="*80)
    print(f"Training data:")
    print(f"  Malicious: {TRAINING_MALICIOUS_DIR}")
    print(f"  Normal:    {TRAINING_NORMAL_DIR}")
    print(f"Output directory: {RESULTS_DIR}")
    print("="*80)
    
    # Define classifiers to train: (name, output_path, use_booleanize)
    classifiers = [
        ("decision-tree", DECISION_TREE_PATH, False),
        ("naive-bayes", NAIVE_BAYES_PATH, True),
        ("svm", SVM_PATH, False),
        ("random-forest", RANDOM_FOREST_PATH, False),
    ]
    
    # Train each classifier
    for classifier_type, output_path, use_boolean in classifiers:
        print(f"\n{'='*80}")
        print(f"Training {classifier_type.upper()} classifier")
        print(f"{'='*80}")
        
        try:
            train_classifier(
                classifier_type=classifier_type,
                malicious_dir=str(TRAINING_MALICIOUS_DIR),
                normal_dir=str(TRAINING_NORMAL_DIR),
                output_path=str(output_path),
                booleanize=use_boolean,
                exclude_features=[],
                nu=0.05,  # Deprecated: Not used for SVC, kept for compatibility
                positive=False,
                render=False,
                render_path=None,
                view=False,
                randomize=False,
                performance_path=str(RESULTS_DIR / 'training_times.csv')
            )
            print(f"[OK] {classifier_type} training completed successfully")
        except Exception as e:
            print(f"[FAIL] {classifier_type} training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("ALL CLASSIFIERS TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Classifiers saved to: {Path(DECISION_TREE_PATH).parent}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

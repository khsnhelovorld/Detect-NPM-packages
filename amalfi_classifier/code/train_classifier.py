#!/usr/bin/env python3
import argparse
import csv
import pickle
import random
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

try:
    from graphviz import Source
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

import sys
from pathlib import Path
code_dir = Path(__file__).parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from read_features import read_training_features, FEATURE_NAMES
from commons import CLASSIFIER_DIR, METRICS_SCORING


def train_classifier(
    classifier_type: str,
    malicious_dir: str,
    normal_dir: str,
    output_path: str,
    booleanize: bool = False,
    exclude_features: list = None,
    nu: float = 0.05,
    positive: bool = False,
    render: bool = False,
    render_path: str = None,
    view: bool = False,
    randomize: bool = False,
    performance_path: str = None
):

    if exclude_features is None:
        exclude_features = []
    
    # Naive Bayes implicitly booleanizes the feature vectors
    if classifier_type == "naive-bayes":
        booleanize = True
    
    print(f"Training {classifier_type} classifier...")
    print(f"  Booleanize: {booleanize}")
    print(f"  Exclude features: {exclude_features}")
    print(f"  Randomize (balance): {randomize}")
    
    # Read features from directories
    feature_vectors, labels, csv_filenames = read_training_features(malicious_dir, normal_dir)
    
    if len(feature_vectors) == 0:
        raise ValueError("No feature vectors loaded. Check directory paths.")
    
    # Process features
    num_samples = len(feature_vectors)
    num_features = len(FEATURE_NAMES)
    
    # Get indices of features to exclude
    exclude_indices = []
    if exclude_features:
        for feat in exclude_features:
            if feat in FEATURE_NAMES:
                exclude_indices.append(FEATURE_NAMES.index(feat))
    
    # Process feature vectors
    processed_vectors = []
    for vec in feature_vectors:
        processed_vec = []
        for i, val in enumerate(vec):
            if i in exclude_indices:
                continue  # Skip excluded features
            
            # Apply positive filter
            if positive and val < 0:
                val = 0.0
            
            # Booleanize
            if booleanize:
                val = 1.0 if val > 0 else 0.0
            
            processed_vec.append(val)
        processed_vectors.append(processed_vec)
    
    # Balance dataset if requested
    if randomize:
        malicious_indices = [i for i, label in enumerate(labels) if label == "malicious"]
        benign_indices = [i for i, label in enumerate(labels) if label == "benign"]
        malicious_count = len(malicious_indices)
        
        # Randomly sample benign samples to match malicious count
        if len(benign_indices) > malicious_count:
            selected_benign = random.sample(benign_indices, malicious_count)
        else:
            selected_benign = benign_indices
        
        # Combine selected samples
        selected_indices = malicious_indices + selected_benign
        processed_vectors = [processed_vectors[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        csv_filenames = [csv_filenames[i] for i in selected_indices]
        
        print(f"Dataset balanced: {len(processed_vectors)} samples ({malicious_count} malicious, {len(selected_benign)} benign)")
    
    # Train classifier
    print(f"Training on {len(processed_vectors)} samples with {len(processed_vectors[0])} features...")
    
    start_time = timer()
    
    if classifier_type == "decision-tree":
        clf = tree.DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_split=20
        )
        clf.fit(processed_vectors, labels)
    elif classifier_type == "random-forest":
        clf = RandomForestClassifier(criterion="entropy", n_estimators=100)
        clf.fit(processed_vectors, labels)
    elif classifier_type == "naive-bayes":
        clf = naive_bayes.BernoulliNB()
        clf.fit(processed_vectors, labels)
    elif classifier_type == "svm":
        # SVC (Supervised SVM) - train on both malicious and benign samples
        # Uses linear kernel with balanced class weights for imbalanced datasets
        # C=1.0 provides standard regularization
        clf = svm.SVC(
            kernel='linear',
            class_weight='balanced',  # Automatically balance classes (malicious/benign)
            probability=True,          # Enable predict_proba for confidence scores
            C=1.0                     # Regularization parameter (higher = less regularization)
        )
        clf.fit(processed_vectors, labels)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    end_time = timer()
    training_time = timedelta(seconds=end_time - start_time)
    
    print(f"Training completed in {training_time}")
    
    # Save performance metrics if requested
    if performance_path:
        with open(performance_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([classifier_type, str(training_time)])
    
    # Render decision tree if requested
    if render and classifier_type == "decision-tree" and render_path:
        if not HAS_GRAPHVIZ:
            print("Warning: graphviz not installed. Cannot render decision tree.")
            print("Install with: pip install graphviz (and install Graphviz binary)")
        elif not render_path.endswith('.png'):
            raise ValueError("Render path must end with .png")
        else:
            try:
                dot_source = tree.export_graphviz(
                    clf,
                    out_file=None,
                    feature_names=[name for i, name in enumerate(FEATURE_NAMES) if i not in exclude_indices],
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = Source(dot_source)
                graph.render(Path(render_path).stem, view=view, cleanup=True)
                print(f"Decision tree rendered to {render_path}")
            except Exception as e:
                print(f"Warning: Could not render decision tree: {e}")
    
    # Save classifier
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get feature names after exclusion
    used_feature_names = [name for i, name in enumerate(FEATURE_NAMES) if i not in exclude_indices]
    
    classifier_data = {
        "feature_names": used_feature_names,
        "all_feature_names": FEATURE_NAMES,
        "excluded_features": exclude_features,
        "booleanize": booleanize,
        "positive": positive,
        "classifier_type": classifier_type,
        "classifier": clf
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(classifier_data, f)
    
    print(f"Classifier saved to {output_file}")
    
    return clf, classifier_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier (Amalfi-style)")
    parser.add_argument(
        "classifier",
        choices=["decision-tree", "random-forest", "naive-bayes", "svm"],
        help="Type of classifier to train"
    )
    parser.add_argument(
        "--malicious-dir",
        type=str,
        help="Directory containing malicious package CSV files"
    )
    parser.add_argument(
        "--normal-dir",
        type=str,
        help="Directory containing normal/benign package CSV files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path to save the pickled classifier"
    )
    parser.add_argument(
        "-b", "--booleanize",
        choices=["true", "false"],
        default="false",
        help="Whether to booleanize feature vectors"
    )
    parser.add_argument(
        "-x", "--exclude-features",
        nargs="*",
        default=[],
        help="List of features to exclude"
    )
    parser.add_argument(
        "-n", "--nu",
        type=float,
        default=0.05,
        help="Deprecated: Previously for OneClassSVM nu parameter. Now using SVC, this parameter is ignored."
    )
    parser.add_argument(
        "-p", "--positive",
        choices=["true", "false"],
        default="false",
        help="Whether to keep only positive values (set negative to 0)"
    )
    parser.add_argument(
        "-r", "--render",
        type=str,
        help="PNG file path to render the decision tree to (only for decision-tree)"
    )
    parser.add_argument(
        "-v", "--view",
        action="store_true",
        help="View the decision tree after rendering"
    )
    parser.add_argument(
        "--randomize",
        choices=["true", "false"],
        default="false",
        help="Balance dataset (randomly sample benign to match malicious count)"
    )
    parser.add_argument(
        "--performance",
        type=str,
        help="CSV file path to log training time performance"
    )
    
    args = parser.parse_args()
    
    # Use default paths if not provided
    try:
        from commons import TRAINING_MALICIOUS_DIR, TRAINING_NORMAL_DIR
        malicious_dir = args.malicious_dir or str(TRAINING_MALICIOUS_DIR)
        normal_dir = args.normal_dir or str(TRAINING_NORMAL_DIR)
    except ImportError:
        # If running as script, add code directory to path
        import sys
        from pathlib import Path
        code_dir = Path(__file__).parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from commons import TRAINING_MALICIOUS_DIR, TRAINING_NORMAL_DIR
        malicious_dir = args.malicious_dir or str(TRAINING_MALICIOUS_DIR)
        normal_dir = args.normal_dir or str(TRAINING_NORMAL_DIR)
    
    train_classifier(
        classifier_type=args.classifier,
        malicious_dir=malicious_dir,
        normal_dir=normal_dir,
        output_path=args.output,
        booleanize=(args.booleanize == "true"),
        exclude_features=args.exclude_features,
        nu=args.nu,
        positive=(args.positive == "true"),
        render=(args.render is not None),
        render_path=args.render,
        view=args.view,
        randomize=(args.randomize == "true"),
        performance_path=args.performance
    )


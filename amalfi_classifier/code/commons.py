#!/usr/bin/env python3

import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, matthews_corrcoef,
    precision_score, recall_score
)


def get_code_dir():
    return Path(__file__).parent.absolute()


def get_project_root():
    return get_code_dir().parent


def get_dataset_root():
    project_root = get_project_root()
    # Navigate from amalfi_classifier -> Workspace -> dataset
    return project_root.parent / 'dataset'


CODE_DIR = get_code_dir()
PROJECT_ROOT = get_project_root()
DATASET_ROOT = get_dataset_root()

# Results directory (created automatically)
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Classifier save directory (created automatically)
CLASSIFIER_DIR = RESULTS_DIR / 'classifiers'
CLASSIFIER_DIR.mkdir(exist_ok=True)

# Visualization output directory (created automatically)
VISUALIZATION_DIR = RESULTS_DIR / 'visualizations'
VISUALIZATION_DIR.mkdir(exist_ok=True)

DECISION_TREE_PATH = CLASSIFIER_DIR / 'decision-tree.pkl'
NAIVE_BAYES_PATH = CLASSIFIER_DIR / 'naive-bayes.pkl'
SVM_PATH = CLASSIFIER_DIR / 'svm.pkl'
RANDOM_FOREST_PATH = CLASSIFIER_DIR / 'random-forest.pkl'

TRAINING_SET_DIR = DATASET_ROOT / 'training_set'
TRAINING_MALICIOUS_DIR = TRAINING_SET_DIR / 'malicious'
TRAINING_NORMAL_DIR = TRAINING_SET_DIR / 'normal'

TEST_SET_DIR = DATASET_ROOT / 'test_set'
TEST_MALICIOUS_DIR = TEST_SET_DIR / 'malicious-dedupli'
TEST_NORMAL_DIR = TEST_SET_DIR / 'normal'

# Scorers for cross-validation and evaluation
METRICS_SCORING = {
    "prec": make_scorer(precision_score, pos_label="malicious"),
    "accu": make_scorer(accuracy_score),
    "rec": make_scorer(recall_score, pos_label="malicious"),
    "f1": make_scorer(f1_score, pos_label="malicious"),
    "matt_cor": make_scorer(matthews_corrcoef)
}

# Field names for CSV output files
METRICS_FIELD_NAMES = ["classifier", "accuracy", "precision", "recall", "f1", "mcc"]

# Number of features in the dataset
NUM_FEATURES = 22

# Default SVM nu parameter (controls the upper bound on fraction of outliers)
DEFAULT_SVM_NU = 0.05

# Random seed for reproducibility (when needed)
RANDOM_SEED = 42
# NT521.Q13.ANTT - Detect malicous npm packages using machine learning
Group3 members:
- Huỳnh Quốc Khánh - 23520718
- Nguyễn Đức Khoa - 23520748
- Võ Ngọc Hoàng Lâm - 23520839
- Đặng Thành Nhân - 23521071

---

A machine learning-based system designed to detect malicious npm packages using static code analysis. This repository implements four distinct classifiers to identify threats based on source code features and metadata.

## Overview

This project adapts the Amalfi research framework to classify npm packages as **Benign** or **Malicious**.

Dataset used can be found at [willTerner](https://github.com/willTerner/detect-malicious-npm-package-with-machine-learning.git). It extracts **22 boolean features** (e.g., presence of network calls, file system access, obfuscation) and trains models to predict the safety of a package.


## Models Implemented

The system trains and compares four classifiers:
1.  **Support Vector Machine (SVM):**
2.  **Random Forest:**
3.  **Naive Bayes:**
4.  **Decision Tree:**

## Installation

Requires Python 3.x and standard data science libraries.

```bash
pip install scikit-learn pandas matplotlib seaborn numpy graphviz
```

## Usage

### 1. Train the Models
Train all four classifiers using the provided dataset.
```bash
cd amalfi_classifier/code
python train.py
```

### 2. Evaluate Performance
Run the test set against the trained models to see Accuracy, Precision, Recall, and F1-Score.
```bash
python evaluate.py
```

### 3. Generate Visualizations
Create confusion matrices and performance comparison charts.
```bash
python visualize.py
```

### 4. Analyze Misclassifications
Identify which packages confuse the models to improve feature engineering.
```bash
python analyze_misclassified.py
```

### 5. Predict a Specific Package
Check a specific package name against the loaded models.
```bash
python predict.py <package_name>
```

## References

[Artifact: Practical Automated Detection of Malicious npm Packages](https://github.com/githubnext/amalfi-artifact.git)
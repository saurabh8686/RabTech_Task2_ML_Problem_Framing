import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# RabTech Academy - Task 2
# ML Problem Framing & Responsible Data Card
# --------------------------------------------------

DATA_PATH = "data/customer-churn-training.csv"

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CUSTOMER CHURN - DATASET ANALYSIS")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["churned"].value_counts())

print("\nTarget Percentage:")
print(
    (df["churned"].value_counts(normalize=True) * 100)
    .round(2)
)

# --------------------------------------------------
# 2. NON-ML MAJORITY CLASS BASELINE
# --------------------------------------------------

y = df["churned"]

majority_class = y.mode()[0]

baseline_predictions = np.full(
    shape=len(y),
    fill_value=majority_class
)

baseline_accuracy = accuracy_score(
    y,
    baseline_predictions
)

baseline_precision = precision_score(
    y,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y,
    baseline_predictions,
    zero_division=0
)

baseline_f1 = f1_score(
    y,
    baseline_predictions,
    zero_division=0
)

baseline_cm = confusion_matrix(
    y,
    baseline_predictions
)

print("\n" + "=" * 70)
print("NON-ML MAJORITY CLASS BASELINE")
print("=" * 70)

print("\nMajority Class:")
print(majority_class)

print("\nBaseline Prediction:")
print(
    "Every customer is predicted as:",
    "Not Churned" if majority_class == 0 else "Churned"
)

print("\nBaseline Accuracy:")
print(f"{baseline_accuracy:.4f}")

print("\nBaseline Precision:")
print(f"{baseline_precision:.4f}")

print("\nBaseline Recall:")
print(f"{baseline_recall:.4f}")

print("\nBaseline F1 Score:")
print(f"{baseline_f1:.4f}")

print("\nBaseline Confusion Matrix:")
print(baseline_cm)

print("\n" + "=" * 70)
print("BASELINE ANALYSIS COMPLETED")
print("=" * 70)
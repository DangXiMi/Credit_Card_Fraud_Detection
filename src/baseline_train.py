import pandas as pd
from preprocessing import load_data, split_time_chronological
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score, precision_score, confusion_matrix

def predict_baseline(X: pd.DataFrame) -> pd.Series:
    """
    Simple rule: flag as fraud if Hour between 0-5 AND Amount >= 100.
    """
    return ((X['Hour'].between(0, 5)) & (X['Amount'] >= 100)).astype(int)

df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_time_chronological(df)

y_pred_zeros = np.zeros(len(y_val))
print("=== All‑zeros baseline (validation) ===")
print("Recall:", recall_score(y_val, y_pred_zeros, zero_division=0))
print("Precision:", precision_score(y_val, y_pred_zeros, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred_zeros))

# Our rule‑based baseline
y_pred_rule = predict_baseline(X_val)
print("\n=== Rule‑based baseline (validation) ===")
print("PR‑AUC:", average_precision_score(y_val, y_pred_rule))
print("Recall:", recall_score(y_val, y_pred_rule, zero_division=0))
print("Precision:", precision_score(y_val, y_pred_rule, zero_division=0))
tn, fp, fn, tp = confusion_matrix(y_val, y_pred_rule).ravel()
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
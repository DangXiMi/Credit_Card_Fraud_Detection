from preprocessing import load_data, split_time_chronological
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, confusion_matrix
import joblib

# 1. Reload data & split
df = load_data()
_, _, X_test, _, _, y_test = split_time_chronological(df)

# 2. Load artefacts
artifacts = joblib.load('models/model_artifacts.pkl')
pipeline = artifacts['pipeline']
threshold = artifacts['threshold']

# 3. Probabilities & predictions
probs_test = pipeline.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= threshold).astype(int)

# 4. Metrics
pr_auc_test = average_precision_score(y_test, probs_test)
recall_test = recall_score(y_test, preds_test, zero_division=0)
precision_test = precision_score(y_test, preds_test, zero_division=0)
f1_test = f1_score(y_test, preds_test, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, preds_test).ravel()

print("=== Test‑Set Evaluation ===")
print(f"PR‑AUC: {pr_auc_test:.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"F1: {f1_test:.4f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
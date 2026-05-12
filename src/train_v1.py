from preprocessing import load_data, split_time_chronological
from pipeline import build_pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, confusion_matrix, recall_score, precision_score
import numpy as np
import joblib

# 1. Load & split
df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_time_chronological(df)

# 2. Scale_pos_weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# 3. Build & train
pipeline = build_pipeline(scale_pos_weight)
pipeline.fit(X_train, y_train)

# 4. Validation probabilities
probs_val = pipeline.predict_proba(X_val)[:, 1]

# 5. Threshold optimisation (max F1)
precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_val, probs_val)
valid_thresholds = thresholds_curve
f1_scores = 2 * (precision_curve[:len(valid_thresholds)] * recall_curve[:len(valid_thresholds)]) / (precision_curve[:len(valid_thresholds)] + recall_curve[:len(valid_thresholds)] + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = valid_thresholds[best_idx]

# 6. Metrics at best threshold
preds_val = (probs_val >= best_threshold).astype(int)
pr_auc = average_precision_score(y_val, probs_val)
recall = recall_score(y_val, preds_val, zero_division=0)
precision = precision_score(y_val, preds_val, zero_division=0)
f1 = f1_score(y_val, preds_val, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_val, preds_val).ravel()

print(f'Validation PR‑AUC: {pr_auc:.4f}')
print(f'Optimal Threshold (F1): {best_threshold:.4f}')
print(f'Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}')
print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')

# 7. Save artefacts
joblib.dump({'pipeline': pipeline, 'threshold': best_threshold, 'val_metrics': {
    'pr_auc': pr_auc, 'recall': recall, 'precision': precision, 'f1': f1
}}, 'models/model_artifacts.pkl')
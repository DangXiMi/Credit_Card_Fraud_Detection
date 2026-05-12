import joblib
import numpy as np
import pandas as pd
import preprocessing as pp
from pipeline import get_pipeline
from config import SETTINGS
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score, precision_score, confusion_matrix

def run_training():
    # Load & prep
    df = pp.load_data(SETTINGS['paths']['raw_data'])
    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_time_chronological(
        df, SETTINGS['train_params']['train_size'], SETTINGS['train_params']['val_size']
    )
    print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    imbalance = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Imbalance ratio (neg/pos) = {imbalance:.4f}")

    # Hyperparameter search
    search = RandomizedSearchCV(
        get_pipeline({}, imbalance),
        SETTINGS['search_space'],
        n_iter=SETTINGS['train_params']['n_iter'],
        scoring='average_precision',
        cv=TimeSeriesSplit(SETTINGS['train_params']['cv_splits']),
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)

    # Validation probabilities
    y_val_proba = search.best_estimator_.predict_proba(X_val)[:, 1]

    # ---------- Threshold selection using max F1 (skip threshold=0 and inf) ----------
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    valid_prec = precisions[:-1]
    valid_rec  = recalls[:-1]
    valid_thresh = thresholds      

    f1_scores = 2 * (valid_prec * valid_rec) / (valid_prec + valid_rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    threshold = valid_thresh[best_idx]

    best_f1 = f1_scores[best_idx]
    best_rec = valid_rec[best_idx]
    best_prec = valid_prec[best_idx]

    print(f"Optimal threshold (max F1): {threshold:.6f}")
    print(f"  → Validation recall = {best_rec:.4f}, precision = {best_prec:.4f}, F1 = {best_f1:.4f}")

    # Retrain on train+val
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    final = get_pipeline(search.best_params_, imbalance)
    final.fit(X_full, y_full)

    # Test evaluation
    y_test_proba = final.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    test_recall = recall_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    test_cm = confusion_matrix(y_test, y_test_pred)

    print("\n=== TEST SET ===")
    print(f"PR-AUC: {test_pr_auc:.4f}")
    print(f"Recall: {test_recall:.4f}  |  Precision: {test_precision:.4f}  |  F1: {2*test_recall*test_precision/(test_recall+test_precision+1e-10):.4f}")
    print("Confusion matrix:")
    print(test_cm)

    # Save artifacts
    # joblib.dump({
    #     'pipeline': final,
    #     'threshold': threshold,
    #     'test_metrics': {
    #         'recall': test_recall,
    #         'precision': test_precision,
    #         'pr_auc': test_pr_auc,
    #         'f1': 2*test_recall*test_precision/(test_recall+test_precision+1e-10)
    #     }
    # }, SETTINGS['paths']['model_output'])
    # print("Done.")

if __name__ == "__main__":
    run_training()
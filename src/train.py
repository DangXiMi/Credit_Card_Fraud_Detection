from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np
import pandas as pd
import preprocessing 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    verbose=True,
    solver='lbfgs',
    class_weight='balanced'
)

# 2. Fix the evaluation function
def evaluate_model(y_true, y_pred_proba):
    """Calculates ROC-AUC and Precision-Recall AUC."""
    auc = roc_auc_score(y_true, y_pred_proba)
    
    pr_auc = average_precision_score(y_true, y_pred_proba) 
    return auc, pr_auc

# 3. Fix naming conventions and logic in threshold function
def find_optimal_threshold(model, X_val, y_val):
    """Finds the threshold that maximizes the F1 score using validation data."""
    thresholds = []
    f1_values = [] 
    
    # Get probabilities once to save computation time
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    for threshold in np.linspace(0.01, 1, 100):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        f1 = f1_score(y_val, y_pred)
        f1_values.append(f1)
        thresholds.append(threshold)
    
    optimal_threshold_idx = np.argmax(f1_values)
    optimal_threshold = thresholds[optimal_threshold_idx]
    optimal_f1 = f1_values[optimal_threshold_idx]
    
    return optimal_threshold, optimal_f1


if __name__ == "__main__":
    # Load and split data
    df = preprocessing.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_time(df)
    
    # Preprocess features (assuming you have this in your preprocessing module)
    X_train, X_val = preprocessing.preprocess_features(X_train, X_val)

    # CRITICAL FIX: Train the model first!
    print("Training the model...")
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set using probabilities
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    auc, pr_auc = evaluate_model(y_val, y_pred_proba_val)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(f"Validation PR-AUC: {pr_auc:.4f}")

    # Find the optimal threshold for F1 score using the VALIDATION set
    optimal_threshold, optimal_f1 = find_optimal_threshold(model, X_val, y_val)
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Best Validation F1 Score: {optimal_f1:.4f}")
    
    # 1. Apply your optimal threshold to the validation probabilities
    y_pred_optimal = (y_pred_proba_val >= optimal_threshold).astype(int)

    # 2. Compute the confusion matrix
    cm = confusion_matrix(y_val, y_pred_optimal)

    # 3. Create the visual plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=['Normal (0)', 'Fraud (1)'])

    # values_format='d' ensures it prints whole numbers instead of scientific notation
    disp.plot(cmap='Blues', values_format='d', ax=ax)

    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.2f})')
    plt.show()
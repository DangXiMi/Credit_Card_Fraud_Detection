import numpy as np

def recall(y_true, y_pred):
    # y_true, y_pred are binary arrays (0/1) of same length
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def precision(y_true, y_pred):
    # implement similarly
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def f1_score(y_true, y_pred):
    # harmonic mean of precision and recall
    rc = recall(y_true,y_pred)
    pr = precision(y_true,y_pred)
    return 2*rc*pr /(rc + pr) if (rc + pr) > 0 else 0.0

def cost_function(y_true, y_pred, fp_cost=10, fn_cost=100):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return fp * fp_cost + fn * fn_cost

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1])    
    print("Recall:", recall(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("f1:", f1_score(y_true, y_pred))
    print("cost_function:", cost_function(y_true, y_pred))
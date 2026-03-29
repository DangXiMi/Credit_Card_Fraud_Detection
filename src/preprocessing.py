# src/preprocess.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(file_path="data/raw/creditcard.csv"):
    return pd.read_csv(file_path)

def split_time_chronological(df):
    """Chronological split: 80% train, 10% val, 10% test based on Time."""
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    
    X_train, y_train = train.drop('Class', axis=1), train['Class']
    X_val, y_val = val.drop('Class', axis=1), val['Class']
    X_test, y_test = test.drop('Class', axis=1), test['Class']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_preprocessor(X_train, columns_to_scale=['Amount', 'Time']):
    """Fit a StandardScaler on the specified columns and return the scaler."""
    scaler = StandardScaler()
    scaler.fit(X_train[columns_to_scale])
    # Save column order (all features)
    column_order = X_train.columns.tolist()
    return scaler, column_order

def transform_features(X, scaler, columns_to_scale=['Amount', 'Time'], column_order=None):
    """Transform Amount & Time using the fitted scaler, then ensure column order."""
    X = X.copy()
    X[columns_to_scale] = scaler.transform(X[columns_to_scale])
    if column_order is not None:
        X = X[column_order]
    return X

def save_preprocessor(scaler, column_order, filepath="models/preprocessor.pkl"):
    """Save scaler and column order together."""
    joblib.dump({'scaler': scaler, 'column_order': column_order}, filepath)

def load_preprocessor(filepath="models/preprocessor.pkl"):
    """Load preprocessor dictionary."""
    return joblib.load(filepath)
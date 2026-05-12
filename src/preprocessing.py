# src/preprocessing.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_data(file_path="data/raw/creditcard.csv"):
    """Stateless: Just reads the file."""
    try:
        df=pd.read_csv(file_path)
        df = df.sort_values('Time').reset_index(drop=True)
        df['Hour'] = (df['Time'] // 3600) % 24
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        raise

def split_time_chronological(df_sorted, train_size=0.8, val_size=0.1):
    """
    Stateless: Logic for data partitioning.
    Includes feature engineering for Time -> Hour to prevent data leakage
    """
    n = len(df_sorted)
    
    train_end = int(train_size * n)
    val_end = int((train_size + val_size) * n)
    
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    
    def get_xy(data):
        return data.drop('Class', axis=1), data['Class']
    
    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# def clean_raw_data(df, log_amount=False):
#     """Stateless: Basic cleaning that doesn't depend on statistics."""
#     df = df.drop_duplicates() 
#     df_sorted = df.sort_values('Time').reset_index(drop=True)
#     df_sorted['Hour'] = (df_sorted['Time'] // 3600) % 24
#     if log_amount:
#         df_sorted['Amount'] = np.log1p(df_sorted['Amount'])
#     df_sorted = df_sorted.drop(columns=['Time'])
#     return df_sorted



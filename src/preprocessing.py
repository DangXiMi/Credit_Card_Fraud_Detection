from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data(file_path="data/raw/creditcard.csv"):
    # Load the credit card dataset from a CSV
    df=pd.read_csv(file_path)
    return df

def split_time(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split 1: 80% Train, 20% Temporary (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Split 2: Split the 20% Temporary set perfectly in half (10% Val, 10% Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_features(X_train, X_test):
    # Apply StandardScaler to Amount and Time columns
    """_
    We have to deal with the Time feature later.
    """
    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    return X_train, X_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_time(df)
    X_train, X_test = preprocess_features(X_train, X_test)
    print(X_train[['Amount', 'Time']])
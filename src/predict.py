# src/predict.py
import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import SETTINGS

app = FastAPI(title="Real-Time Credit Card Fraud API", version="1.0")

MODEL_PATH = SETTINGS['paths']['model_output']
OPTIMAL_THRESHOLD = 0.1751

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Error: {e}")

class TransactionIn(BaseModel):
    Time: float
    Amount: float
    model_config = ConfigDict(extra='allow')

@app.post("/predict")
def predict_fraud(transaction: TransactionIn):
    try:
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])
        
        if df.shape[1] < 30:
            raise ValueError("Missing PCA features (V1-V28).")

        df['Hour'] = (df['Time'] // 3600) % 24
        df = df.drop(columns=['Time'])
        
        fraud_prob = pipeline.predict_proba(df)[0, 1]
        
        is_fraud = int(fraud_prob >= OPTIMAL_THRESHOLD)
        
        return {
            "transaction_status": "Fraudulent" if is_fraud else "Legitimate",
            "fraud_probability": round(float(fraud_prob), 4),
            "is_fraud": bool(is_fraud),
            "action": "BLOCK" if is_fraud else "APPROVE"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)
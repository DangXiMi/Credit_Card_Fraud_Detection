# src/predict.py
import os, sys, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import SETTINGS

app = FastAPI(title="Real-Time Fraud Detection API", version="2.0")

# Load artifacts (pipeline + threshold)
artifacts = joblib.load(SETTINGS['paths']['model_output'])
pipeline = artifacts['pipeline']
THRESHOLD = artifacts['threshold']

class Transaction(BaseModel):
    features: List[float] = Field(..., min_items=30, max_items=30, description="30 features: V1..V28, Amount, Time")
    
    def to_dataframe(self):
        # Convert to DataFrame with correct column names
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame([self.features], columns=cols)
        # Feature engineering (same as training)
        df['Hour'] = (df['Time'] // 3600) % 24
        df = df[cols + ['Hour']]
        return df

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        df = transaction.to_dataframe()
        proba = pipeline.predict_proba(df)[0, 1]
        is_fraud = int(proba >= THRESHOLD)
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud": bool(is_fraud),
            "action": "BLOCK" if is_fraud else "APPROVE",
            "threshold_used": THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=False)
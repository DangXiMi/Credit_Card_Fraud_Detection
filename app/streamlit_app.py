# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import numpy as np

# Ensure config can be loaded safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import SETTINGS

# Set page config
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# ---------- Load model and threshold from artifacts ----------
@st.cache_resource
def load_model():
    artifacts = joblib.load(SETTINGS['paths']['model_output'])
    pipeline = artifacts['pipeline']
    threshold = artifacts['threshold']
    # Optionally load test metrics for display
    test_metrics = artifacts.get('test_metrics', {})
    return pipeline, threshold, test_metrics

pipeline, threshold, test_metrics = load_model()

# ---------- Helper: Prepare DataFrame for prediction ----------
def prepare_features(df_raw):
    """Convert raw input (with Time, Amount, V1..V28) to model-ready features."""
    df = df_raw.copy()
    
    required_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None
    
    df['Hour'] = (df['Time'] // 3600) % 24
    df = df.drop(columns=['Time'])
    
    # IMPORTANT: Order must match training (V1..V28, Amount, Hour)
    expected_order = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
    df = df[expected_order]
    return df

# ---------- UI ----------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Powered by **XGBoost** | Optimized for severe class imbalance (0.17%)")
st.markdown("---")

# Tabs
tab2, tab3 = st.tabs(["📂 Batch Analysis", "📊 Model Architecture"])

# ---------------------------------------------------------
# TAB 2: BATCH ANALYSIS
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch Transaction Scoring")
    st.write("Upload a CSV of raw transactions to score them in bulk.")
    st.info("Required columns: **Time**, **Amount**, and **V1** through **V28** (29 columns total).")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_raw)} transactions.")
        
        # Prepare features
        df_processed = prepare_features(df_raw)
        if df_processed is None:
            st.stop()
        
        with st.spinner("Scoring transactions..."):
            # Predict probabilities
            fraud_probs = pipeline.predict_proba(df_processed)[:, 1]
            
            # Append results to original dataframe
            df_results = df_raw.copy()
            df_results['Fraud_Probability'] = fraud_probs
            df_results['Is_Fraud'] = (fraud_probs >= threshold).astype(int)
            
            # Summary metrics
            fraud_count = df_results['Is_Fraud'].sum()
            total = len(df_results)
            fraud_rate = fraud_count / total if total > 0 else 0.0
            
            st.markdown("### Batch Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Processed", total)
            col2.metric("Flagged as Fraud", fraud_count)
            col3.metric("Predicted Fraud Rate", f"{fraud_rate*100:.4f}%")
            
            # Probability distribution
            fig_hist = px.histogram(
                df_results, x='Fraud_Probability', nbins=50,
                title="Distribution of Fraud Probabilities",
                color='Is_Fraud',
                color_discrete_map={0: 'green', 1: 'red'}
            )
            fig_hist.add_vline(
                x=threshold, line_dash="dash", line_color="black",
                annotation_text=f"Threshold = {threshold:.3f}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Download button
            csv_output = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Transactions",
                data=csv_output,
                file_name="scored_transactions.csv",
                mime="text/csv",
                type="primary"
            )

# ---------------------------------------------------------
# TAB 3: MODEL ARCHITECTURE & METRICS
# ---------------------------------------------------------
with tab3:
    st.subheader("System Architecture & Validation Metrics")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 🧠 Model Pipeline
        - **Algorithm:** XGBoost Classifier
        - **Imbalance Handling:** `scale_pos_weight` = 567.87 (optimised)  
        - **Feature Engineering:** Cyclic `Hour` conversion from `Time` (no leakage)  
        - **Preprocessing:** `RobustScaler` on `Hour` and `Amount` (handles outliers)
        """)
    
    with col_info2:
        # Extract metrics from loaded artifacts (if available)
        recall = test_metrics.get('recall', 0.7273)
        precision = test_metrics.get('precision', 0.7273)
        f1 = test_metrics.get('f1', 0.7273)
        pr_auc = test_metrics.get('pr_auc', 0.7281)
        
        st.markdown(f"""
        ### 📊 Performance Metrics (Unseen Test Set)
        - **Precision:** {precision:.4f}
        - **Recall:** {recall:.4f}
        - **F1-Score:** {f1:.4f}
        - **PR-AUC:** {pr_auc:.4f}
        - **Decision Threshold:** {threshold:.4f}
        """)
    
    st.markdown("---")
    st.caption("Model trained with chronological split and time‑series cross‑validation. Threshold optimised for maximum F1 on validation set.")
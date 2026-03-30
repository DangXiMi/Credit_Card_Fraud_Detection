import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Ensure config can be loaded safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import SETTINGS

# Set page config for a professional wide layout
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# Cache the model loading so it doesn't reload on every button click
@st.cache_resource
def load_model():
    pipeline = joblib.load(SETTINGS['paths']['model_output'])
    # Hardcoded from our Phase 4 dynamic tuning
    threshold = 0.1751 
    return pipeline, threshold

pipeline, threshold = load_model()

# Header Section
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Powered by **XGBoost** & **RobustScaler** | Optimized for severe class imbalance (0.17%)")
st.markdown("---")

# Tabs for organization
tab2, tab3 = st.tabs(["📂 Batch Analysis", "📊 Model Architecture"])


# ---------------------------------------------------------
# TAB 2: BATCH ANALYSIS
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch Transaction Scoring")
    st.write("Upload a CSV of raw transactions to score them in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(df_raw)} transactions.")
        
        with st.spinner("Scoring transactions..."):
            df_processed = df_raw.copy()
            
            # Safely apply Feature Engineering if 'Time' exists
            if 'Time' in df_processed.columns:
                df_processed['Hour'] = (df_processed['Time'] // 3600) % 24
                df_processed = df_processed.drop(columns=['Time'])
            
            # Drop target column if it was accidentally uploaded
            if 'Class' in df_processed.columns:
                df_processed = df_processed.drop(columns=['Class'])
            
            # Predict probabilities
            fraud_probs = pipeline.predict_proba(df_processed)[:, 1]
            
            # Append results to the original dataframe for the user
            df_results = df_raw.copy()
            df_results['Fraud_Probability'] = fraud_probs
            df_results['Is_Fraud'] = (fraud_probs >= threshold).astype(int)
            
            # Display Summary Metrics
            fraud_count = df_results['Is_Fraud'].sum()
            
            st.markdown("### Batch Summary")
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("Total Processed", len(df_results))
            b_col2.metric("Flagged as Fraud", fraud_count)
            b_col3.metric("Fraud Rate", f"{(fraud_count/len(df_results))*100:.2f}%")
            
            # Distribution Plot
            fig_hist = px.histogram(
                df_results, x='Fraud_Probability', nbins=50, 
                title="Distribution of Fraud Probabilities",
                color='Is_Fraud', color_discrete_map={0: 'green', 1: 'red'}
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Decision Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Download Button
            csv_output = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Transactions",
                data=csv_output,
                file_name="scored_transactions.csv",
                mime="text/csv",
                type="primary"
            )

# ---------------------------------------------------------
# TAB 3: MODEL INFO
# ---------------------------------------------------------
with tab3:
    st.subheader("System Architecture & Validation Metrics")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        ### 🧠 Model Pipeline
        * **Algorithm:** XGBoost Classifier
        * **Imbalance Handling:** `scale_pos_weight` & `max_delta_step`
        * **Feature Engineering:** Cyclic `Hour` conversion to prevent data leakage.
        * **Scaling:** `RobustScaler` to resist extreme transaction amount outliers.
        """)
        
    with info_col2:
        st.markdown("""
        ### 📊 Performance Metrics (Unseen Test Set)
        * **Precision (Fraud):** 0.70
        * **Recall (Fraud):** 0.73
        * **F1-Score:** 0.71
        * **Dynamic Threshold:** 0.1751 (Optimized via Validation F1 maximization)
        """)